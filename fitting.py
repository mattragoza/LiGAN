import molgrid
from rdkit.Chem import AllChem as Chem
from openbabel import pybel, openbabel
import torch
import numpy as np
import seaborn as sns
from rdkit.Geometry.rdGeometry import Point3D
from skimage.segmentation import flood_fill
from generate import *
from atom_types import *
from collections import namedtuple
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def grid_to_xyz(gcoords, mgrid):
    return mgrid.center+(np.array(gcoords)-((mgrid.size-1)/2))*mgrid.resolution

def select_atom_starts(mgrid, G, radius):
    '''Given a single channel grid and the atomic radius for that type,
    select initial positions using a weight random selection that treats 
    each disconnected volume of density separately'''
    per_atom_volume = radius**3*((2*np.pi)**1.5) 

    mask = G.cpu().numpy()
    values = mask.copy()
    
    #look for islands of density greater than 0.5 (todo: parameterize this threshold?)
    #label each island in mask
    THRESHOLD = 0.5
    mask[mask >= THRESHOLD] = 1.0
    mask[mask < THRESHOLD] = 0
    
    maxpos = np.unravel_index(mask.argmax(),mask.shape)
    masks = []
    which = -1
    while mask[maxpos] > 0:
        flood_fill(mask, maxpos, which, in_place=True) #identify and mark the connected region
        maxpos = np.unravel_index(mask.argmax(),mask.shape)
        which -= 1            

    for selector in range(-1,which,-1):
         masks.append(mask == selector)

    retcoords = []

    #print("#masks",len(masks))
    for M in masks:
        maskedG = G.cpu().numpy()
        maskedG[~M] = 0
        flatG = maskedG.flatten()
        total = flatG.sum()
        cnt = int(np.ceil(float(total)/per_atom_volume))  #pretty sure this can only underestimate
        #counting this way is especially problematic for large molecules that go to the box edge
        if cnt == 0:
            continue
       
        flatG[flatG > 1.0] = 1.0
        rand = np.random.choice(range(len(flatG)), cnt, False, flatG/flatG.sum())
        gcoords = np.array(np.unravel_index(rand,G.shape)).T
        ccoords = grid_to_xyz(gcoords, mgrid)
        
        retcoords += list(ccoords)

    #print("coords",len(retcoords))
    return retcoords   
       
    
def simple_atom_fit(mgrid, types,iters=10,tol=0.01,device='cuda'):
    '''Fit atoms to MolGrid.  types are ignored as the number of 
    atoms of each type is always inferred from the density.
    Returns the MolGrid of the placed atoms and the MolStruct'''

    #for every channel, select some coordinates and setup the type/radius vectors
    initcoords = []
    typevecs = []
    radii = []
    typeindices = []    
    numatoms = 0
    tcnts = {}
    for (t,G) in enumerate(mgrid.values):
        ch = mgrid.channels[t]
        coords = select_atom_starts(mgrid, G, ch.atomic_radius)
        if coords:    
            tvec = np.zeros(len(mgrid.channels))
            tvec[t] = 1.0
            tcnt = len(coords)
            numatoms += tcnt
            
            r = mgrid.channels[t].atomic_radius
            initcoords += coords
            typevecs += [tvec]*tcnt
            typeindices += [t]*tcnt
            radii += [r]*tcnt
            tcnts[t] = tcnt
            
    typevecs = np.array(typevecs)
    initcoords = np.array(initcoords)
    typeindices = np.array(typeindices)

    #setup gridder
    gridder = molgrid.Coords2Grid(molgrid.GridMaker(dimension=mgrid.dimension,resolution=mgrid.resolution,
                                                    gaussian_radius_multiple=-1.5),center=mgrid.center)
    mgrid.values = mgrid.values.to(device)

    #having setup input coordinates, optimize with BFGS
    coords = torch.tensor(initcoords,dtype=torch.float32,requires_grad=True,device=device)
    types = torch.tensor(typevecs,dtype=torch.float32,device=device)
    radii = torch.tensor(radii,dtype=torch.float32,device=device)
    best_loss = np.inf
    best_coords = None
    best_typeindices = typeindices #save in case number of atoms changes
    goodcoords = False
    
    for inum in range(iters):
        optimizer = torch.optim.LBFGS([coords],max_iter=20000,tolerance_grad=1e-9,line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            agrid = gridder.forward(coords,types,radii)   
            loss = torch.square(agrid-mgrid.values).sum()/numatoms
            loss.backward()
            return loss

        optimizer.step(closure)
        final_loss = optimizer.state_dict()['state'][0]['prev_loss']  #todo - check for convergence?
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_coords = coords.detach().cpu()
            
        if inum == iters-1: #stick with these coordinates
            break;
        #otherwise, try different starting coordinates for only those
        #atom types that have errors
        goodcoords = True
        with torch.no_grad():
            offset = 0
            agrid = gridder.forward(coords,types,radii)     
            t = 0
            while offset < len(typeindices):
                t = typeindices[offset]
                #eval max error - mse will downplay a single atom of many being off
                maxerr = float(torch.square(agrid[t]-mgrid.values[t]).max())
                if maxerr > tol:
                    goodcoords = False
                    ch = mgrid.channels[t]
                    newcoords = select_atom_starts(mgrid, mgrid.values[t], ch.atomic_radius)
                    for (i,coord) in enumerate(newcoords):
                        coords[i+offset] = torch.tensor(coord,dtype=torch.float)
                offset += tcnts[t]
        if goodcoords:
            break                            
        
    numfixes = 0
    if not goodcoords:
        #try to fix up an atom at a time
        offset = 0
        #reset corods to best found so far
        with torch.no_grad():
            coords[:] = best_coords
            agrid = gridder.forward(coords,types,radii)     
        t = 0
        while offset < len(typeindices):
            t = typeindices[offset]
            maxerr = float(torch.square(agrid[t]-mgrid.values[t]).max())
            per_atom_volume = float(radii[offset])**3*((2*np.pi)**1.5) 
            while maxerr > tol:
                #identify the atom of this type closest to the place with too much density
                #and move it to the location with too little density
                tcoords = coords[offset:offset+tcnts[t]].detach().cpu().numpy() #coordinates for this type
                
                diff = agrid[t]-mgrid.values[t]
                possum = float(diff[diff>0].sum())
                negsum = float(diff[diff <0].sum())
                maxdiff = float(diff.max())
                mindiff = float(diff.min())
                missing_density = -(negsum+possum)
                #print('Type %d numcoords %d maxdiff %.5f mindiff %.5f missing %.5f'%(t,len(tcoords),maxdiff,mindiff,missing_density))
                if missing_density > .25*per_atom_volume: #add atom  MAGIC NUMBER ALERT
                    #needs to be enough total missing density to be close to a whole atom,
                    #but the missing density also needs to be somewhat concentrated
                    #print("Missing density - not enough atoms?")
                    numfixes += 1
                    minpos = int((agrid[t]-mgrid.values[t]).argmin())
                    minpos = grid_to_xyz(np.unravel_index(minpos,agrid[t].shape),mgrid)
                    #add atom: change coords, types, radii, typeindices and tcnts, numatoms
                    numatoms += 1
                    typeindices = np.insert(typeindices, offset, t)
                    tcnts[t] += 1
                    with torch.no_grad():
                        newcoord = torch.tensor([minpos],device=coords.device,dtype=coords.dtype,requires_grad=True)
                        coords = torch.cat((coords[:offset],newcoord,coords[offset:]))
                        radii = torch.cat((radii[:offset],radii[offset:offset+1],radii[offset:]))
                        types = torch.cat((types[:offset],types[offset:offset+1],types[offset:]))
                        
                        coords.requires_grad_(True)
                        radii.requires_grad_(True)
                        types.requires_grad_(True)
                    
                elif missing_density < -.75*per_atom_volume:
                    print("Too many atoms?")
                    break
                    #todo, remove atom
                else:   #move an atom
                    numfixes += 1
                    maxpos = int((agrid[t]-mgrid.values[t]).argmax())
                    minpos = int((agrid[t]-mgrid.values[t]).argmin())
                    maxpos = grid_to_xyz(np.unravel_index(maxpos,agrid[t].shape),mgrid)
                    minpos = grid_to_xyz(np.unravel_index(minpos,agrid[t].shape),mgrid)

                    dists = np.square(tcoords - maxpos).sum(axis=1)
                    closesti = np.argmin(dists)
                    with torch.no_grad():
                        coords[offset+closesti] = torch.tensor(minpos)
                
                #reoptimize               
                optimizer = torch.optim.LBFGS([coords],max_iter=20000,tolerance_grad=1e-9,line_search_fn='strong_wolfe')
                #TODO: only optimize this grid
                optimizer.step(closure)
                final_loss = optimizer.state_dict()['state'][0]['prev_loss']  #todo - check for convergence?
                agrid = gridder.forward(coords,types,radii) #recompute grid
                
                #if maxerr hasn't improved, give up
                newerr = float(torch.square(agrid[t]-mgrid.values[t]).max())
                #print(t,'newerr',newerr,'maxerr',maxerr,'maxdiff',maxdiff,'missing',missing_density)
                if newerr >= maxerr:
                    #don't give up if there's still a lot left to fit
                    #and the missing density isn't all shallow
                    if missing_density < per_atom_volume or maxdiff < 0.01: #magic number! 
                        break
                else:
                    maxerr = newerr
                    best_loss = final_loss
                    best_coords = coords.detach().cpu()
                    best_typeindices = typeindices.copy()

                #otherwise update coordinates and repeat
                
            offset += tcnts[t]
            
    #create struct from coordinates
    mol = MolStruct(best_coords.numpy(), best_typeindices, mgrid.channels)
    #print('losses',final_loss,best_loss)
    return (float(best_loss),inum,numfixes,mol)


def fixup(atoms, mol, struct):
    '''Set atom properties to match channel.  Keep doing this
    to beat openbabel over the head with what we want to happen.'''
    mol.SetAromaticPerceived(True)  #avoid perception
    for atom,t in zip(atoms,struct.c):
        ch = struct.channels[t]
        if 'Aromatic' in ch.name:
            atom.SetAromatic(True)
            atom.SetHyb(2)

        if 'Donor' in ch.name:
            if atom.GetExplicitDegree() == atom.GetHvyDegree():
                if atom.GetHvyDegree() == 1 and atom.GetAtomicNum() == 7:
                    atom.SetImplicitHCount(2)
                else:
                    atom.SetImplicitHCount(1) 

                    
        elif 'Acceptor' in ch.name: # NOT AcceptorDonor because of else
            atom.SetImplicitHCount(0)   
        
        if ('Nitrogen' in ch.name or 'Oxygen' in ch.name) and atom.IsInRing(): 
            #this is a little iffy, ommitting until there is more evidence it is a net positive
            #we don't have aromatic types for nitrogen, but if it
            #is in a ring with aromatic carbon mark it aromatic as well
            acnt = 0
            for nbr in openbabel.OBAtomAtomIter(atom):
                if nbr.IsAromatic():
                    acnt += 1
            if acnt > 1:
                atom.SetAromatic(True)
         
                            
        
def reachable_r(a,b, seenbonds):
    '''Recursive helper.'''

    for nbr in openbabel.OBAtomAtomIter(a):
        bond = a.GetBond(nbr).GetIdx()
        if bond not in seenbonds:
            seenbonds.add(bond)
            if nbr == b:
                return True
            elif reachable_r(nbr,b,seenbonds):
                return True
    return False
        
def reachable(a,b):
    '''Return true if atom b is reachable from a without using the bond between them.'''
    if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:
        return False #this is the _only_ bond for one atom
    #otherwise do recursive traversal
    seenbonds = set([a.GetBond(b).GetIdx()])
    return reachable_r(a,b,seenbonds)

def forms_small_angle(a,b,cutoff=45):
    '''Return true if bond between a and b is part of a small angle
    with a neighbor of a only.'''

    for nbr in openbabel.OBAtomAtomIter(a):
        if nbr != b:
            degrees = b.GetAngle(a,nbr)
            if degrees < cutoff:
                return True
    return False
    
def connect_the_dots(mol, atoms, struct, maxbond=4):
    '''Custom implementation of ConnectTheDots.  This is similar to
    OpenBabel's version, but is more willing to make long bonds 
    (up to maxbond long) to keep the molecule connected.  It also 
    attempts to respect atom type information from struct.
    atoms and struct need to correspond in their order
    
    Assumes no hydrogens or existing bonds.
    '''
    mol.BeginModify()


    #just going to to do n^2 comparisons, can worry about efficiency later
    coords = np.array([(a.GetX(),a.GetY(),a.GetZ()) for a in atoms])
    dists = squareform(pdist(coords))
    types = [struct.channels[t].name for t in struct.c]
    
    for (i,a) in enumerate(atoms):
        for (j,b) in enumerate(atoms):
            if a == b:
                break
            if dists[i,j] < 0.4:
                continue #don't bond too close atoms
            if dists[i,j] < maxbond:
                flag = 0
                if 'Aromatic' in types[i] and 'Aromatic' in types[j]:
                    flag = openbabel.OB_AROMATIC_BOND
                mol.AddBond(a.GetIdx(),b.GetIdx(),1,flag)

    #cleanup = remove long bonds
    atom_maxb = {}
    for (i,a) in enumerate(atoms):
        maxb = openbabel.GetMaxBonds(a.GetAtomicNum()) #don't exceed this
        if 'Donor' in types[i]:
            maxb -= 1 #leave room for hydrogen
        atom_maxb[a.GetIdx()] = maxb
    
    bonds = [b for b in openbabel.OBMolBondIter(mol)]
    binfo = []
    for bond in bonds:
        bdist = bond.GetLength()
        #compute how far away from optimal we are
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        ideal = openbabel.GetCovalentRad(a1.GetAtomicNum()) + openbabel.GetCovalentRad(a2.GetAtomicNum()) 
        stretch = bdist-ideal
        binfo.append((stretch,bdist,bond))
    binfo.sort(reverse=True) #most stretched bonds first
    
    for stretch,bdist,bond in binfo:
        #can we remove this bond without disconnecting the molecule?
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        #don't fragment the molecule
        if not reachable(a1,a2):
            continue

        #as long as we aren't disconnecting, let's remove things
        #that are excessively far away (0.45 from ConnectTheDots)
        #get bonds to be less than max allowed
        #also remove tight angles, because that is what ConnectTheDots does
        if stretch > 0.45 or  \
            a1.GetExplicitValence() > atom_maxb[a1.GetIdx()] or \
            a2.GetExplicitValence() > atom_maxb[a2.GetIdx()] or \
            forms_small_angle(a1,a2) or forms_small_angle(a2,a1):
            mol.DeleteBond(bond)
            continue                        
            
    mol.EndModify()
            
    

def make_obmol(struct,verbose=False):
    '''Create an OBMol from MolStruct that attempts to maintain
    correct atom typing'''
    mol = openbabel.OBMol()
    mol.BeginModify()
    
    
    atoms = []
    for xyz,t in zip(struct.xyz, struct.c):
        x,y,z = map(float,xyz)
        ch = struct.channels[t]
        atom = mol.NewAtom()
        atom.SetAtomicNum(ch.atomic_num)
        atom.SetVector(x,y,z)            
        atoms.append(atom)
        
    fixup(atoms, mol, struct)    
    connect_the_dots(mol, atoms, struct)
    fixup(atoms, mol, struct)
    mol.EndModify()

    mol.AddPolarHydrogens() #make implicits explicit
    
    mol.PerceiveBondOrders()
    fixup(atoms, mol, struct)
    
    for (i,a) in enumerate(atoms):
        openbabel.OBAtomAssignTypicalImplicitHydrogens(a)

    fixup(atoms, mol, struct)
    
    mol.AddHydrogens()
    fixup(atoms, mol, struct)
    mismatches = 0
    for (a,t) in zip(atoms,struct.c):
        ch = struct.channels[t]
        if 'Donor' in ch.name and not a.IsHbondDonor():
            mismatches += 1
            if verbose:
                print("Not Donor",ch.name,a.GetX(),a.GetY(),a.GetZ())
        if ch.name != 'NitrogenXSDonorAcceptor' and 'Acceptor' in ch.name and a.GetExplicitDegree() != a.GetTotalDegree():
            #there are issues with nitrogens and openbabel protonation..
            mismatches += 1
            if verbose:
                print("Not Acceptor",ch.name,a.GetX(),a.GetY(),a.GetZ())
            
        if 'Aromatic' in ch.name and not a.IsAromatic():
            mismatches += 1
            if verbose:
                print("Not Aromatic",ch.name,a.GetX(),a.GetY(),a.GetZ())
        

    return pybel.Molecule(mol),mismatches
    
def calc_valence(rdatom):
    '''Can call GetExplicitValence before sanitize, but need to
    know this to fix up the molecule to prevent sanitization failures'''
    cnt = 0.0
    for bond in rdatom.GetBonds():
        cnt += bond.GetBondTypeAsDouble()
    return cnt
    
def convert_ob_mol_to_rd_mol(ob_mol):
    '''Convert OBMol to RDKit mol, fixing up issues'''
    ob_mol.DeleteHydrogens()
    n_atoms = ob_mol.NumAtoms()
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    for ob_atom in ob.OBMolAtomIter(ob_mol):
        rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
        #TODO copy format charge
        if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:
            #don't commit to being aromatic unless rdkit will be okay with the ring status
            #(this can happen if the atoms aren't fit well enough)
            rd_atom.SetIsAromatic(True)        
        i = rd_mol.AddAtom(rd_atom)
        ob_coords = ob_atom.GetVector()
        x = ob_coords.GetX()
        y = ob_coords.GetY()
        z = ob_coords.GetZ()
        rd_coords = Geometry.Point3D(x, y, z)
        rd_conf.SetAtomPosition(i, rd_coords)

    rd_mol.AddConformer(rd_conf)

    nonsingles = []
    positions = rd_mol.GetConformer().GetPositions()
    
    for ob_bond in ob.OBMolBondIter(ob_mol):
        i = ob_bond.GetBeginAtomIdx()-1
        j = ob_bond.GetEndAtomIdx()-1
        bond_order = ob_bond.GetBondOrder()
        if bond_order == 1:
            rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
        elif bond_order == 2:
            rd_mol.AddBond(i, j, Chem.BondType.DOUBLE)
        elif bond_order == 3:
            rd_mol.AddBond(i, j, Chem.BondType.TRIPLE)          
        else:
            raise Exception('unknown bond order {}'.format(bond_order))
            
        if ob_bond.IsAromatic():
            bond = rd_mol.GetBondBetweenAtoms (i,j)
            bond.SetIsAromatic(True)    
            
            
    rd_mol = Chem.RemoveHs(rd_mol, sanitize=False)
                
    pt = Chem.GetPeriodicTable()
    #if double/triple bonds are connected to hypervalent atoms, decrement the order
    nosingles = []
    for bond in rd_mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetBondType() == Chem.BondType.TRIPLE:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            dist = np.linalg.norm(positions[i]-positions[j])
            nonsingles.append((dist,bond))
    nonsingles.sort(reverse=True)
    
    for (d,bond) in nonsingles:
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        
        if calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
           calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):
            btype = Chem.BondType.SINGLE
            if bond.GetBondType() == Chem.BondType.TRIPLE:
                btype = Chem.BondType.DOUBLE
            bond.SetBondType(btype)  
            
    #set nitrogens with 4 neighbors to have a charge
    for atom in rd_mol.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
            atom.SetFormalCharge(1)
            
    rd_mol = Chem.AddHs(rd_mol,addCoords=True)
    #Kekulize will lose our aromatic flags :-()

    Chem.SanitizeMol(rd_mol,Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
        
    #but at some point stop trying to enforce our aromaticity -
    #openbabel and rdkit have different aromaticity models so they
    #won't always agree.  Remove any aromatic bonds to non-aromatic atoms
    for bond in rd_mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if bond.GetIsAromatic():
            if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                bond.SetIsAromatic(False)
        elif a1.GetIsAromatic() and a2.GetIsAromatic():
            bond.SetIsAromatic(True)
    
    return rd_mol
    
def make_rdmol(struct,verbose=False):
    '''Create RDKIT mol from MolStruct trying to respect types.'''
    mol,misses = make_mol(struct,verbose)
    return ob_mol_to_rd_mol(mol.OBMol)
