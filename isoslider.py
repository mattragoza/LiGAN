'''
Isocontour slider ("density slider") plugin for PyMOL

See also this similar plugin:
http://www.ebi.ac.uk/~gareth/pymol/downloads/scripts/density_slider.py

(c) 2013 Thomas Holder

License: BSD-2-Clause
'''

from __future__ import print_function

try:
    import Tkinter
except ImportError:
    import tkinter as Tkinter

from pymol import cmd, plugins

DIGITS = 2
DELTA = 10 ** (-DIGITS)


def __init_plugin__(self=None):
    plugins.addmenuitem('Isocontour Slider', isoslider)


def get_isoobjects(state=1, quiet=1):
    '''
    Get a list of (name, isolevel) tuples for all isomesh and isosurface
    objects.
    '''
    state, quiet = int(state), int(quiet)
    r = []
    for name in cmd.get_names():
        t = cmd.get_type(name)
        if t in ('object:mesh', 'object:surface'):
            level = cmd.isolevel(name, 0, state, 1)
            if not quiet:
                print('%-20s %5.2f' % (name, level))
            r.append((name, level))
    return r


class LevelVar(Tkinter.Variable):

    '''
    Tk variable that is bound to an isocontour object
    '''

    def __init__(self, master, name, level):
        Tkinter.Variable.__init__(self, master, value=level)
        self.name = name
        self.trace('w', self.callback)

    def callback(self, *args):
        cmd.isolevel(self.name, self.get())

    def increment(self, event=None, delta=DELTA):
        self.set(round(float(self.get()) + delta, DIGITS))

    def decrement(self, event=None):
        self.increment(None, -DELTA)

    def bindscrollwheel(self, element):
        element.bind('<Button-4>', self.increment)
        element.bind('<Button-5>', self.decrement)


class GroupLevelVar(Tkinter.Variable):

    '''
    Tk variable that is bound to an isocontour object
    '''

    def __init__(self, master, names, level):
        Tkinter.Variable.__init__(self, master, value=level)
        self.names = names
        self.trace('w', self.callback)

    def callback(self, *args):
        for name in self.names:
            cmd.isolevel(name, self.get())

    def increment(self, event=None, delta=DELTA):
        self.set(round(float(self.get()) + delta, DIGITS))

    def decrement(self, event=None):
        self.increment(None, -DELTA)

    def bindscrollwheel(self, element):
        element.bind('<Button-4>', self.increment)
        element.bind('<Button-5>', self.decrement)


def isoslider(mm_min=0.0, mm_max=1.0):
    '''
    DESCRIPTION

    Opens a dialog with isolevel sliders for all isomesh and isosurface
    objects in PyMOL.
    '''
    mm_min = float(mm_min)
    mm_max = float(mm_max)
    top = Tkinter.Toplevel(plugins.get_tk_root())
    master = Tkinter.Frame(top, padx=5, pady=5)
    master.pack(fill="both", expand=1)
    mmvar = Tkinter.DoubleVar(top, value=mm_min)

    def fillmaster():
        ffmt = '{:.' + str(DIGITS) + 'f}'
        for child in list(master.children.values()):
            child.destroy()

        mm = mmvar.get()
        mmf = Tkinter.Frame(master)
        Tkinter.Label(mmf, text=ffmt.format(mm_min)).grid(row=0, column=0, sticky='w')
        Tkinter.Label(mmf, text=ffmt.format((mm_max + mm_min)/2)).grid(row=0, column=1)
        Tkinter.Label(mmf, text=ffmt.format(mm_max)).grid(row=0, column=2, sticky='e')
        mmf.grid(row=0, column=1, sticky='ew')
        mmf.columnconfigure(1, weight=1)

        names = []
        for i, (name, level) in enumerate(get_isoobjects(), 1):
            names.append(name)
            continue

            v = LevelVar(master, name, ffmt.format(level))
            Tkinter.Label(master, text=name).grid(row=i+2, column=0, sticky="w")
            e = Tkinter.Scale(master, orient=Tkinter.HORIZONTAL,
                              from_=mm_min, to=mm_max, resolution=DELTA,
                              showvalue=0, variable=v)
            e.grid(row=i+2, column=1, sticky="ew")
            v.bindscrollwheel(e)
            e = Tkinter.Entry(master, textvariable=v, width=6)
            e.grid(row=i+2, column=2, sticky="e")
            v.bindscrollwheel(e)
            master.columnconfigure(1, weight=1)

        v = GroupLevelVar(master, names, ffmt.format(mm_min))
        Tkinter.Label(master, text='all').grid(row=1, column=0, sticky="w")
        e = Tkinter.Scale(master, orient=Tkinter.HORIZONTAL,
                          from_=mm_min, to=mm_max, resolution=DELTA,
                          showvalue=0, variable=v)
        e.grid(row=1, column=1, sticky="ew")
        v.bindscrollwheel(e)
        e = Tkinter.Entry(master, textvariable=v, width=6)
        e.grid(row=1, column=2, sticky="e")
        v.bindscrollwheel(e)
        master.columnconfigure(1, weight=1)

    fillmaster()

    bottom = Tkinter.Frame(top, padx=5)
    Tkinter.Label(bottom, text="+/-").pack(side=Tkinter.LEFT)
    mmentry = Tkinter.Entry(bottom, textvariable=mmvar, width=6)
    mmentry.pack(side=Tkinter.LEFT)
    refresh = Tkinter.Button(bottom, text="Refresh", command=fillmaster)
    refresh.pack(side=Tkinter.LEFT)
    bottom.pack(side=Tkinter.BOTTOM, fill="x", expand=1)

cmd.extend('isoslider', isoslider)
