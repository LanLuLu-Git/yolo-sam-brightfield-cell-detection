import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# 确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


def show_anns(anns):
    """可视化分割掩码"""
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    # 按面积从大到小排序
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    for ann in sorted_anns:
        m = ann['segmentation']
        # 随机生成掩码颜色
        color_mask = np.random.random(3)
        # 将掩码转换为彩色并叠加
        img = np.ones((m.shape[0], m.shape[1], 3))
        img[:, :, 0] = color_mask[0]
        img[:, :, 1] = color_mask[1]
        img[:, :, 2] = color_mask[2]
        ax.imshow(np.dstack((img, m * 0.4)))


def segment_multi_objects(image_path, model_type='vit_h', checkpoint_path='D:/Project/YOLOwithSAM/segment-anything-main/models/sam_vit_h_4b8939.pth', output_path=None):
    """
    使用SAM模型分割图像中的多个物体

    参数:
        image_path: 输入图像路径
        model_type: 模型类型 ('vit_h', 'vit_l', 'vit_b')
        checkpoint_path: 模型权重路径
        output_path: 输出图像保存路径，为None则显示图像
    """
    # 加载模型
    print("正在加载模型...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

    # 创建掩码生成器
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=15,  # 采样点数
        pred_iou_thresh=0.90,  # IOU阈值
        stability_score_thresh=0.95,  # 稳定性分数阈值
        crop_n_layers=1,  # 裁剪的层数。这个参数控制了模型在生成掩码时是否对图像进行裁剪以及裁剪的层数。设置为 1 表示只进行一层裁剪。
        crop_n_points_downscale_factor=2,  # 裁剪时采样点数的下采样因子。
        # 这个参数控制了在裁剪后的图像上采样的点数相对于原始图像的比例。
        # 设置为 2 表示在裁剪后的图像上采样的点数是原始图像的一半。
        min_mask_region_area=40000,  # 过滤小区域
    )

    # 加载图像
    print("正在加载图像...")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 生成掩码
    print("正在分割图像中的物体...")
    masks = mask_generator.generate(image)
    print(f"已检测到 {len(masks)} 个物体")

    # 可视化结果
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.title("原始图像")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(122)
    plt.title(f"已分割 {len(masks)} 个物体")
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')

    plt.tight_layout()

    # 保存或显示结果
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"结果已保存至: {output_path}")
    else:
        plt.show()

    return masks


if __name__ == "__main__":
    # 用户需修改此处的图像路径
    input_image = "D:/Project/YOLOwithSAM/ultralytics-main/ultralytics/assets/bus.jpg"

    # 可选: 指定输出图像路径，取消注释下面一行
    # output_image = "segmented_result.jpg"

    try:
        masks = segment_multi_objects(
            image_path=input_image,
            model_type='vit_h',  # 可选: 'vit_l', 'vit_b'
            checkpoint_path='D:/Project/YOLOwithSAM/segment-anything-main/models/sam_vit_h_4b8939.pth'
            # output_path=output_image  # 取消注释此行以保存结果
        )
    except Exception as e:
        print(f"处理过程中出错: {e}")