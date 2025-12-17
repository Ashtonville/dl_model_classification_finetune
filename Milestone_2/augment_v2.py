import os
from pathlib import Path
from torchvision.transforms import transforms
from PIL import Image
from os.path import isfile, join, isdir

root_dir = "./data/train/"
csv_file = "./data/train.csv"

new_root_dir = "data/aug_train_v2/"
new_csv_file = "data/aug_train_v2.csv"

transform_resize = transforms.Compose([transforms.Resize((224,224)),])

transform_vit_safe = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.1,
        hue=0.05
    ),
    transforms.RandomRotation(
        degrees=10,
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill=0
    ),
    transforms.RandomErasing(
        p=0.15,
        scale=(0.01, 0.04),
        ratio=(0.3, 3.3),
        value="random"
    ),

    transforms.ToPILImage(),
])

transform_vit_structure = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomAffine(
        degrees=8,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05),
        shear=5,
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.2,
        saturation=0.1,
        hue=0.03
    ),
    transforms.ToPILImage(),
])

transform_vit_strong = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.25,
        saturation=0.2,
        hue=0.08
    ),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomRotation(
        degrees=15,
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.RandomErasing(
        p=0.2,
        scale=(0.02, 0.06),
        ratio=(0.5, 2.0),
        value="random"
    ),
    transforms.ToPILImage(),
])


with open(new_csv_file, 'w', newline='') as csvfile:
    csvfile.write("image:FILE,category\n")
    i = 0
    for plant_type_dir in next(os.walk(root_dir))[1]:
        Path("/".join([new_root_dir, plant_type_dir])).mkdir(parents=True, exist_ok=True)
        for file in next(os.walk(root_dir + plant_type_dir))[2]:
            filename = "/".join([plant_type_dir,file])
            img = Image.open(root_dir + filename)

            img_resize = transform_resize(img)
            line = ",".join([filename,str(i)])+"\n"
            csvfile.write(line)
            img_resize.save("/".join([new_root_dir, filename]))

            abc = ["a","b"]

            for letter in abc:
                if letter == "a":
                    transform_augment = transform_vit_safe
                if letter == "b":
                    transform_augment = transform_vit_structure
                else:
                    transform_augment = transform_vit_strong

                img_augmented = transform_augment(img_resize)

                x = file.split(".")
                x.insert(1, letter)
                augmented_filename = ".".join(x)
                augmented_filename = "/".join([plant_type_dir,augmented_filename])
                line = ",".join([augmented_filename,str(i)]) + "\n"
                csvfile.write(line)
                img_augmented.save("/".join([new_root_dir, augmented_filename]))

                print(f'Augmented Image: {filename}')
        
        i += 1


