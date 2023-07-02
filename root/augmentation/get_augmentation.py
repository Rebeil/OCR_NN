from torch import manual_seed

manual_seed(17)
import torchvision as tv


def aug():
    h_w = (100, 100)
    h_w_2 = (90, 90)
    l = [float(i) for i in range(0, 250, 25)]
    augmenters = {
        'Crop': tv.transforms.Compose([
            tv.transforms.Resize(size=127, max_size=128,
                                 interpolation=tv.transforms.InterpolationMode.BILINEAR),
            tv.transforms.CenterCrop(size=h_w),
            tv.transforms.RandomCrop(size=h_w),
        ]),
        'Rotate': tv.transforms.RandomRotation(degrees=(-90, 90)),
        # 'HFlip': tv.transforms.RandomHorizontalFlip(p=1),
        # 'VFlip': tv.transforms.RandomVerticalFlip(p=0.5),
        'BlurGauss': tv.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        'R_and_Cp': tv.transforms.Compose([
            tv.transforms.Resize(size=127, max_size=128),
            tv.transforms.RandomRotation(degrees=(-90, 90)),
            tv.transforms.CenterCrop(size=h_w),
        ]),
        'Crop2': tv.transforms.Compose([
            tv.transforms.Resize(size=127, max_size=128,
                                 interpolation=tv.transforms.InterpolationMode.BILINEAR),
            tv.transforms.CenterCrop(size=h_w_2),
            tv.transforms.RandomCrop(size=h_w_2),
        ]),
        'Elastic': tv.transforms.ElasticTransform(alpha=(50.0, 150.0), sigma=5.0),
        'Elastic2': tv.transforms.ElasticTransform(alpha=(50.0, 250.0), sigma=5.0),
        'AugMix': tv.transforms.AugMix()
    }
