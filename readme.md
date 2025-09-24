# Usage

## 1. 安装依赖
推荐使用 [pixi](https://pixi.sh/latest/installation/) 创建项目环境, 安装完成后进入该项目目录，然后执行  
```bash
pixi install    # 根据 pixi.lock 安装依赖
```
安装完成后，执行
```bash
pixi shell
```
进入项目环境.

如果使用conda, 或是 pixi 安装时出现问题，可以参考 `conda_environment.yml` 安装依赖。

## 配置 XCATV2

[XCAT](https://cvit.duke.edu/resource/xcat-anatomy-files/) 是由杜克大学开发的，用于生成 CT 体模的工具，也可以模拟心脏跳动。详细介绍可参考其[论文](https://aapm.onlinelibrary.wiley.com/doi/full/10.1118/1.3480985)

默认将 XCATV2_V2_LINUX 解压到 `./XCAT`，并将 XCATV2 附带的 nrb 模型文件放在 `./xcat_adult_nrb_files` 目录下，如果安装在了其他地方，可以使用 `XCAT_HOME` 与 `XCAT_ADULT_NRB_FILES` 环境变量指定安装位置, 或是在后续运行相关命令时在命令行中指定。

## 运行

使用 `pixi shell` 或 `conda activate` 进入环境后，可使用 `python main.py <task>` 运行指定命令， 也可以用 `python main.py --help` 或 `python main.py <task> --help` 查看相关帮助（第一次启动可能耗时较长）， 支持的命令`<task>`有：

1. generate-ssm-data: 使用 XCAT 及默认或给定的所有 nrb 模型文件生成对应的 4D 心脏体模跳动 CTA 文件，以NII格式保存(时间较长, 临时文件在 <output_dir>/temp中)
2. get-surface-cloud: 从命令1所生成的体模中提取表面点云，以 VTK 格式保存，用于生成模版点云、点云间配准和计算统计形状模型
3. get-volume-cloud: 从命令1所生成的体模文件中体积点云，以 VTK 格式保存，用于配准过程以提配准高精度
4. align-surface: 使用表面点云做病人间配准，并将模板点云作为landmark再次配准到所有实例上，从而得到一一对应的点云集，以 VTK 格式保存
5. calculate-ssm： 使用配准后的点云集计算统计形状模型，结果为numpy矩阵，以npy格式保存

具体的运行命令可参见 `documentation.md` 文件。

计算 SSM 的方法参考自 [TT-UNET](https://github.com/ZihengD/TT-U-Net) 和 [4D Statistical Atlas](https://link.springer.com/chapter/10.1007/11566489_50)

## 其他

脚本 `generate_heartbeat_ssm/_06_generate_4d_image.py` 用于测试使用 SSM 用于真实图像标签的效果，功能在 `Gen4D` 中已全部实现，故暂时已废弃

另有一些脚本在 `script` 下，是设计程序时遗留的，基本已废弃
