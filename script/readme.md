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

这可能是因为我是用female 作为模板，导致有较大差异，无法通过弹性配准对齐

