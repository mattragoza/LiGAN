import sys, os, time, glob
import pandas as pd
from collections import namedtuple
import molgrid
from rdkit.Chem import AllChem as Chem
from openbabel import pybel, openbabel
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Geometry.rdGeometry import Point3D
from skimage.segmentation import flood_fill
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import generate
import atom_types


idx = [2, 3, 4, 5, 19, 18, 17, 6, 9, 7, 8, 10, 13, 12, 16, 14, 15, 20, 27]
channels = atom_types.get_channels_by_index(idx) #generate.py defaults
typer = molgrid.SubsettedGninaTyper(idx,False) #equivalent in molgrid
gmaker = molgrid.GridMaker(gaussian_radius_multiple=-1.5,dimension=36)

device = 'cuda'


def grid_to_xyz(gcoords, mgrid):
    return mgrid.center+(np.array(gcoords)-((mgrid.size-1)/2))*mgrid.resolution


def get_per_atom_volume(radius):
    return radius**3*((2*np.pi)**1.5)  


def select_atom_starts(mgrid, G, radius):
    '''Given a single channel grid and the atomic radius for that type,
    select initial positions using a weight random selection that treats 
    each disconnected volume of density separately'''
    per_atom_volume = get_per_atom_volume(radius)   

    mask = G.cpu().numpy()
    mask[mask > 0] = 1.0
    maxpos = np.unravel_index(mask.argmax(),mask.shape)
    masks = []
    while mask[maxpos] > 0:
        flood_fill(mask, maxpos, -1, in_place=True) #identify and mark the connected region
        masks.append(mask == -1) # save the boolean selctor for this region
        mask[mask == -1] = 0 # remove it from the binary mask
        maxpos = np.unravel_index(mask.argmax(),mask.shape)

    retcoords = []

    for M in masks:
        maskedG = G.cpu().numpy()
        maskedG[~M] = 0
        flatG = maskedG.flatten()
        total = flatG.sum()
        cnt = int(np.ceil(float(total)/per_atom_volume))  #pretty sure this can only underestimate
        #counting this way is especially problematic for large molecules that go to the box edge
        
        flatG[flatG > 1.0] = 1.0
        rand = np.random.choice(range(len(flatG)), cnt, False, flatG/flatG.sum())
        gcoords = np.array(np.unravel_index(rand,G.shape)).T
        ccoords = grid_to_xyz(gcoords, mgrid)
        
        retcoords += list(ccoords)

    return retcoords
       
    
def simple_atom_fit(mgrid,types,iters=10,tol=0.01):
    '''Fit atoms to MolGrid.  types are ignored as the number of 
    atoms of each type is always inferred from the density.
    Returns the MolGrid of the placed atoms and the MolStruct'''
    t_start = time.time()

    mgrid = generate.MolGrid(
        values=torch.as_tensor(mgrid.values, device=device),
        channels=mgrid.channels,
        center=tuple(mgrid.center.astype(float)),
        resolution=mgrid.resolution,
    )

    #for every channel, select some coordinates and setup the type/radius vectors
    initcoords = []
    typevecs = []
    radii = []
    typeindices = []    
    numatoms = 0
    tcnts = {}
    types_est = []
    for (t,G) in enumerate(mgrid.values):
        ch = mgrid.channels[t]
        types_est.append(G.sum()/get_per_atom_volume(ch.atomic_radius))
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

    types_true = torch.tensor(types,dtype=torch.float32,device=device)
    types_est = torch.tensor(types_est,dtype=torch.float32,device=device)
    print(types_est, flush=True)

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

        print(
            'iter {} (loss={}, n_atoms={})'.format(inum, final_loss, len(best_typeindices))
        )
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_coords = coords.detach()
            
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
    fix_iter = 0
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
                if missing_density > .75*per_atom_volume: #add atom
                    print("Missing density - not enough atoms?")
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

                    
                elif mindiff**2 < tol:
                    print("No significant density underage - too many atoms?")
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
                fix_iter += 1
                print(
                    'fix_iter {} (loss={}, n_atoms={}, newerr={}, numfixes={})'.format(
                        fix_iter, final_loss, len(typeindices), newerr, numfixes
                    )
                )

                if newerr >= maxerr:
                    break
                else:
                    maxerr = newerr
                    best_loss = final_loss
                    best_coords = coords.detach()
                    best_typeindices = typeindices.copy()

                #otherwise update coordinates and repeat
                
            offset += tcnts[t]

    n_atoms = len(best_typeindices)
    n_channels = len(mgrid.channels)
    best_types = torch.zeros((n_atoms, n_channels), dtype=torch.float32, device=device)
    best_radii = torch.zeros((n_atoms,), dtype=torch.float32, device=device)
    for i, t in enumerate(best_typeindices):
        ch = mgrid.channels[t]
        best_types[i,t] = 1.0
        best_radii[i] = ch.atomic_radius

    #create struct and grid from coordinates
    grid_best = generate.MolGrid(
        values=gridder.forward(best_coords,best_types,best_radii).cpu().detach().numpy(),
        channels=mgrid.channels,
        center=mgrid.center,
        resolution=mgrid.resolution)

    struct_best = generate.MolStruct(
        xyz=best_coords.cpu().numpy(),
        c=best_typeindices,
        channels=mgrid.channels,
        loss=float(best_loss),
        type_diff=(types_est - best_types.sum(dim=0)).abs().sum().item(),
        est_type_diff=(types_true - types_est).abs().sum().item(),
        time=time.time()-t_start,
        n_steps=numfixes,
    )

    return grid_best, struct_best


def fixup(atoms, mol, struct):
    '''Set atom properties to match channel.  Keep doing this
    to beat openbabel over the head with what we want to happen.'''
    mol.SetAromaticPerceived(True)  #avoid perception
    for atom,t in zip(atoms,struct.c):
        ch = struct.channels[t]
        if 'Aromatic' in ch.name:
            atom.SetAromatic(True)
            for nbr in openbabel.OBAtomAtomIter(atom):
                if nbr.IsAromatic():
                    bond = atom.GetBond(nbr)
                    bond.SetAromatic()
        if 'Donor' in ch.name:
            if atom.GetTotalDegree() - atom.GetHvyDegree() <= 0: 
                atom.SetImplicitHCount(1) # this is nice in theory, but connectthedots is going to ignore implicit valances
        elif 'Acceptor' in ch.name: # NOT AcceptorDonor because of else
            atom.SetImplicitHCount(0)   
        
        if False and 'Nitrogen' in ch.name and atom.IsInRing(): 
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
                    flag = ob.OB_AROMATIC_BOND
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
            

def make_mol(struct,verbose=False):
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
            
    mol.PerceiveBondOrders()
    fixup(atoms, mol, struct)

    
    for (i,a) in enumerate(atoms):
        openbabel.OBAtomAssignTypicalImplicitHydrogens(a)

    fixup(atoms, mol, struct)
            
    mol.EndModify()
    
    mol.AddHydrogens()
    fixup(atoms, mol, struct)
    
    mismatches = 0
    for (a,t) in zip(atoms,struct.c):
        ch = struct.channels[t]
        if 'Donor' in ch.name and not a.IsHbondDonor():
            mismatches += 1
            if verbose:
                print("Not Donor",ch.name,a.GetX(),a.GetY(),a.GetZ())
        if ch.name != 'NitrogenXSDonorAcceptor' and 'Acceptor' in ch.name and not a.IsHbondAcceptorSimple():
            #there are issues with nitrogens and openbabel protonation..
            mismatches += 1
            if verbose:
                print("Not Acceptor",ch.name,a.GetX(),a.GetY(),a.GetZ())
            
        if 'Aromatic' in ch.name and not a.IsAromatic():
            mismatches += 1
            if verbose:
                print("Not Aromatic",ch.name,a.GetX(),a.GetY(),a.GetZ())
        

    return pybel.Molecule(mol),mismatches


def fitmol(fname,niters=10):
    print('Reading {}'.format(fname))
    m = next(pybel.readfile('sdf',fname))
    m.OBMol.Center()  #put in center of box!
    m.addh()
    ligname = os.path.split(fname)[1]
    print('Typing input molecule')
    cset = molgrid.CoordinateSet(m,typer)
    print('Creating empty grid')
    mgrid_values = torch.zeros(gmaker.grid_dimensions(cset.num_types()),dtype=torch.float32,device=device)
    print('Calling gmaker forward')
    gmaker.forward((0,0,0),cset,mgrid_values)

    mgrid = generate.MolGrid(mgrid_values, channels, (0,0,0), 0.5)

    start = time.time()
    loss, fixes, struct = simple_atom_fit(mgrid, None,niters)
    fittime = time.time()-start
    
    try:
        rmsd = get_min_rmsd(cset.coords, cset.type_index.tonumpy(), struct.xyz, struct.c)
    except:
        rmsd = np.inf;
            
    return struct, fittime, loss, fixes, rmsd


if __name__ == '__main__':      
    results = []

    print('Globbing input files')
    files = glob.glob('/net/pulsar/home/koes/dkoes/PDBbind/refined-set/*/*_lig.sdf')

    print('Starting to fit molecules')
    for (i,fname) in enumerate(files):
        try:
            start = time.time()
            struct, fittime, loss, fixes, rmsd = fitmol(fname,25)
            mol,misses = make_mol(struct)
            
            totaltime = time.time()-start
            ligname = os.path.split(fname)[1]    

            mol.write('sdf','output/fit_%s'%ligname,overwrite=True)
            print('{}/{}'.format(i+1, len(files)))        
        except IndexError as e:
            print("Failed",fname,e)


    results = pd.DataFrame(results,columns=('lig','loss','fixes','fittime','totaltime','misses','rmsd'))
    results.to_csv('cntfixes.csv')

    sns.boxplot(data=results,x='misses',y='loss')
    plt.savefig('loss_by_misses_box.png')

    plt.hist(results.loss,bins=np.logspace(-6,1,8))
    plt.gca().set_xscale('log')
    plt.savefig('loss_hist.png')

    print('Low loss but nonzero misses, sorted by misses:')
    print(results[(results.loss < 0.1) & (results.misses > 0)].sort_values(by='misses'))

    print('Overall average loss:')
    print(np.mean(results.loss))

    plt.hist(results.fittime)
    plt.savefig('fit_time_hist.png')

    print('Average fit time and total time')
    print(np.mean(results.fittime))
    print(np.mean(results.totaltime))

    print('Undefined RMSD sorted by loss:')
    print(results[np.isinf(results.rmsd)].sort_values(by='loss'))

    print('All results sorted by loss:')
    print(results.sort_values(by='loss'))
