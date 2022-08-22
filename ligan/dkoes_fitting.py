import time, os
import molgrid
from rdkit.Chem import AllChem as Chem
from openbabel import openbabel as ob
from openbabel import pybel
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
from rdkit import Geometry
from skimage.segmentation import flood_fill
from collections import namedtuple
import pickle

from .atom_structs import AtomStruct


def grid_to_xyz(gcoords, mgrid):
    return mgrid.center+(np.array(gcoords)-((mgrid.size-1)/2))*mgrid.resolution


def select_atom_starts(mgrid, G, radius):
    '''Given a single channel grid and the atomic radius for that type,
    select initial positions using a weight random selection that treats 
    each disconnected volume of density separately'''
    per_atom_volume = radius**3*((2*np.pi)**1.5) 

    mask = G.cpu().numpy().copy()

    #look for islands of density greater than 0.5 (todo: parameterize this threshold?)
    #label each island in mask
    THRESHOLD = 0.5
    values = G.cpu().numpy()
    mask[values >= THRESHOLD] = 1.0
    mask[values < THRESHOLD] = 0

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
        total = float(flatG.sum())

        if total < .1*per_atom_volume:
            continue #should be very conservative given a 0.5 THRESHOLD
        cnt = int(np.ceil(total/per_atom_volume))  #pretty sure this can only underestimate
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


def simple_atom_fit(mgrid, types,iters=10,tol=0.01,device='cuda',grm=-1.5):
    '''Fit atoms to AtomGrid.  types are ignored as the number of 
    atoms of each type is always inferred from the density.
    Returns the AtomGrid of the placed atoms and the AtomStruct'''

    t_start = time.time()
    #for every channel, select some coordinates and setup the type/radius vectors
    initcoords = []
    typevecs = []
    radii = []
    typeindices = []
    numatoms = 0
    tcnts = {}
    values = torch.tensor(mgrid.values,device=device)

    for (t,G) in enumerate(values):
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
    #print('typeindices',typeindices)
    #setup gridder
    center = tuple([float(c) for c in mgrid.center])
    gridder = molgrid.Coords2Grid(molgrid.GridMaker(dimension=mgrid.dimension,resolution=mgrid.resolution,
                                                    gaussian_radius_multiple=grm),center=center)

    #having setup input coordinates, optimize with BFGS
    coords = torch.tensor(initcoords,dtype=torch.float32,requires_grad=True,device=device)
    types = torch.tensor(typevecs,dtype=torch.float32,device=device)
    radii = torch.tensor(radii,dtype=torch.float32,device=device)
    best_loss = np.inf
    best_coords = None
    best_typeindices = typeindices #save in case number of atoms changes
    goodcoords = False
    bestagrid =  torch.zeros(values.shape,dtype=torch.float32,device=device)

    if len(initcoords) == 0: #no atoms
        mol = AtomStruct(np.zeros((0,3)),np.zeros(0), mgrid.channels,
            L2_loss=values.square().sum()/values.numel(),
            time=time.time()-t_start,
            iterations=0,
            numfixes=0,
            type_diff=0,
            est_type_diff=0,
            visited_structs=[]
        )
        return mol, bestagrid

    for inum in range(iters):
        optimizer = torch.optim.LBFGS([coords],max_iter=20000,tolerance_grad=1e-9,line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            agrid = gridder.forward(coords,types,radii)
            loss = torch.square(agrid-values).sum()/numatoms
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
                maxerr = float(torch.square(agrid[t]-values[t]).max())
                if maxerr > tol:
                    goodcoords = False
                    ch = mgrid.channels[t]
                    newcoords = select_atom_starts(mgrid, values[t], ch.atomic_radius)
                    for (i,coord) in enumerate(newcoords):
                        coords[i+offset] = torch.tensor(coord,dtype=torch.float)
                offset += tcnts[t]
        if goodcoords:
            break
    bestagrid = agrid.clone()
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
            maxerr = float(torch.square(agrid[t]-values[t]).max())
            #print('maxerr',maxerr)
            per_atom_volume = float(radii[offset])**3*((2*np.pi)**1.5) 
            while maxerr > tol:
                #identify the atom of this type closest to the place with too much density
                #and move it to the location with too little density
                tcoords = coords[offset:offset+tcnts[t]].detach().cpu().numpy() #coordinates for this type

                diff = agrid[t]-values[t]
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
                    minpos = int((agrid[t]-values[t]).argmin())
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
                    maxpos = int((agrid[t]-values[t]).argmax())
                    minpos = int((agrid[t]-values[t]).argmin())
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
                newerr = float(torch.square(agrid[t]-values[t]).max())
                #print(t,'newerr',newerr,'maxerr',maxerr,'maxdiff',maxdiff,'mindiff',mindiff,'missing',missing_density)
                if newerr >= maxerr:
                    #don't give up if there's still a lot left to fit
                    #and the missing density isn't all (very) shallow
                    if missing_density < per_atom_volume or mindiff > -0.1: #magic number! 
                        break
                else:
                    maxerr = newerr
                    best_loss = final_loss
                    best_coords = coords.detach().cpu()
                    best_typeindices = typeindices.copy()
                    bestagrid = agrid.clone()

                #otherwise update coordinates and repeat

            offset += tcnts[t]

    #create struct from coordinates
    mol = AtomStruct(best_coords.numpy(), best_typeindices, mgrid.channels,
            L2_loss=float(best_loss),
            time=time.time()-t_start,
            iterations=inum,
            numfixes=numfixes,
            type_diff=0,
            est_type_diff=0,
            visited_structs=[])
    # print('losses',final_loss,best_loss,len(best_coords))
    return mol,bestagrid
