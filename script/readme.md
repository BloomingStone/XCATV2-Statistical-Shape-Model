注意这里所用的pycpd或者torchcpd都是修改过的，原有代码无法支持

生成 surface 时有两例(female_pt76, male_pt200)因为生成的surface不是manifold的， 故排除

对齐时因为
```
    self.W = th.linalg.solve(A, B)
             ^^^^^^^^^^^^^^^^^^^^^
torch._C._LinAlgError: torch.linalg.solve: The solver failed because the input matrix is singular.
```
失败的例子有

male_pt108, male_pt118, male_pt128, male_pt141, male_pt146, male_pt154, male_pt159, male_pt168, male_pt184, 共9个

PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt141/ssm_001.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/female_pt117/ssm_002.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt144/ssm_003.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt154/ssm_005.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt144/ssm_006.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt144/ssm_007.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt128/ssm_007.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt128/ssm_009.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt154/ssm_010.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt128/ssm_011.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt184/ssm_012.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt108/ssm_013.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt154/ssm_013.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt96/ssm_014.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt144/ssm_014.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt144/ssm_016.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/female_pt140/ssm_016.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt184/ssm_017.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt141/ssm_017.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt128/ssm_017.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt141/ssm_018.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt144/ssm_018.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt141/ssm_019.vtk'), PosixPath('/media/data3/sj/Data/Phatom/output_ssm_vtk/male_pt154/ssm_019.vtk')

这可能是因为我是用female 作为模板，导致有较大差异，无法通过弹性配准对齐

TODO 这是因为图像边缘被裁切到了

![alt text](image.png)

后面尝试增加fps到20

生成nii图像时 female 142 失败

生成surface时 male 173 失败 （not manifold）