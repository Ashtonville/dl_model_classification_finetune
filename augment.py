from torchvision.transforms import transforms
from PIL import Image
from os import listdir
from os.path import isfile, join

root_dir = "./data/cat_dog"
csv_file = "./data/cat_dog.csv"

new_root_dir = "./new_data/cat_dog"
new_csv_file = "./new_data/cat_dog.csv"


transform_resize = transforms.Compose([transforms.Resize((224,224)),])

transform_augment = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomGrayscale(p=0.3),
    transforms.RandomErasing(p=0.3, scale=(0.02,0.1),ratio=(1,1)),
    transforms.RandomApply([
        transforms.RandomRotation(degrees=60),
    ], p=0.3),
    transforms.RandomApply([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    ], p=0.3),
    transforms.ToPILImage(),
])

files = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]

with open(new_csv_file, 'w', newline='') as csvfile:
    csvfile.write("image,labels\n")
    for index, f in enumerate(files):
        img = Image.open(join(root_dir, f))
        img_resize = transform_resize(img)
        img_augmented = transform_augment(img_resize)

        resize_filename = f

        x = f.split(".")
        x.insert(1, "a")
        augmented_filename = ".".join(x)

        isDog = int(x[0] == "dog")

        csvfile.write(resize_filename + "," + str(isDog) + "\n")
        csvfile.write(augmented_filename + "," + str(isDog) + "\n")

        img_resize.save(join(new_root_dir, resize_filename))
        img_augmented.save(join(new_root_dir, augmented_filename))

        print(f'Augmented Image: {index} of {len(files)}')


