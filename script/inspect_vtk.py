from pathlib import Path
import pyvista as pv
import random
import argparse


d = Path("/media/data3/sj/Data/Phatom/output_ssm_vtk")
vtk_file_list = list(d.rglob("*.vtk"))
random_vtk_file = random.choice(vtk_file_list)
pv.read(random_vtk_file).plot()
