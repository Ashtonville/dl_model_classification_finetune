import os
from pathlib import Path
from torchvision.transforms import transforms
from PIL import Image
from os.path import isfile, join, isdir

root_dir = "./data/train/"
csv_file = "./data/train.csv"

new_root_dir = "./new_train/"
new_csv_file = "./new_train/train.csv"

transform_resize = transforms.Compose([transforms.Resize((224,224)),])

#AUGMENTATION LOSS
transform_augment_loss = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((224, 224)),
    transforms.RandomChoice(
        [transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3)]
    ),
    transforms.RandomGrayscale(p=0.3),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1),ratio=(1,1)),
    transforms.RandomApply([
        transforms.RandomRotation(degrees=20),],
        p=0.3
    ),
    transforms.ToPILImage(),
])

# AUGMENTATION BLUR
transform_augment_blur = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((224, 224)),
    transforms.RandomChoice(
        [transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3)]
    ),
    transforms.RandomApply([
        transforms.ColorJitter((0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2)),
    ], p=0.3),
    transforms.RandomApply([transforms.ElasticTransform(alpha=(50.0, 250.0), sigma=(5.0, 15.0))],
                           p=0.3),
    transforms.RandomApply([
        transforms.RandomRotation(degrees=20),
    ], p=0.3),
    
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
                    transform_augment = transform_augment_loss
                else:
                    transform_augment = transform_augment_blur

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


