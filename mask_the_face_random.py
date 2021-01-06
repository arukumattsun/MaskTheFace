import argparse
import glob

import dlib

from utils.aux_functions import *

MASK_TYPE_LIST = ['surgical_green', 'surgical_blue', 'N95', 'cloth', 'KN95']
TEXTURE_LIST = ['check1', 'check2', 'check3', 'check4', 'check5', 'check6', 'check7']
COLOR_LIST = ['cyan_blue', 'dimgray', 'pink', 'olive']

COLOR_CODE_DICT = {
    'cyan_blue': '#0473e2',
    'dimgray': '#696969',
    'pink': '#ffc0cb',
    'olive': '#808000'
}

TEXTURE_PATH_DICT = {
    'check1': 'masks/textures/check/check_1.jpg',
    'check2': 'masks/textures/check/check_2.jpg',
    'check3': 'masks/textures/check/check_3.jpg',
    'check4': 'masks/textures/check/check_4.jpg',
    'check5': 'masks/textures/check/check_5.jpg',
    'check6': 'masks/textures/check/check_6.jpg',
    'check7': 'masks/textures/check/check_7.jpg',

}


def parse_args():
    parser = argparse.ArgumentParser(
        description="MaskTheFace - Python code to mask faces dataset"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="data/images/",
        help="Path to either the folder containing images or the image itself",
    )

    return parser.parse_args()


class MaskArgument(object):
    def __init__(self, path_to_dlib_model):
        super(MaskArgument, self).__init__()

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path_to_dlib_model)

        self.mask_type = 'N95'
        self.verbose = True

        self.code = False
        self.mask_dict_of_dict = {}

        self.pattern = ''
        self.pattern_weight = 0.0

        self.color = '#0473e2'
        self.color_weight = 0.0

    def set_mask_type_random(self):
        if random.random() < 0.7:
            self.mask_type = 'surgical'
        else:
            self.mask_type = random.choice(MASK_TYPE_LIST)

    def set_mask_color_random(self, use_texture=True):
        if self.mask_type == 'surgical':
            rand = random.random()
            if rand < 0.3:
                self.color = COLOR_CODE_DICT[random.choice(COLOR_LIST)]
                self.color_weight = random.uniform(0.4, 0.6)
                self.pattern = ''
            elif rand < 0.35 and use_texture:
                self.pattern = TEXTURE_PATH_DICT[random.choice(TEXTURE_LIST)]
                self.pattern_weight = random.uniform(0.4, 0.6)
            else:
                self.color_weight = 0.0
                self.pattern_weight = 0.0
                self.pattern = ''
        else:
            self.color_weight = 0.0
            self.pattern_weight = 0.0
            self.pattern = ''


def main():
    args = parse_args()

    is_directory, is_file, is_other = check_path(args.path)

    if is_directory:
        image_paths = [p for p in glob.glob(os.path.join(args.path, '**/*.jpg'), recursive=True)]
        out_dir = args.path + '_mask'
        out_paths = [p.replace(args.path, out_dir) for p in image_paths]
    elif is_file:
        image_paths = [args.path]
        file_name, ext = os.path.splitext(os.path.basename(args.path))
        out_name = file_name + '_mask'
        out_paths = [args.path.replace(file_name, out_name)]
    else:
        print("Path is neither a valid file or a valid directory")
        exit()

    path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
    mask_args = MaskArgument(path_to_dlib_model)
    for i, in_path in tqdm(enumerate(image_paths)):
        mask_args.set_mask_type_random()
        mask_args.set_mask_color_random()
        try:
            masked_image, mask, mask_binary_array, original_image = mask_image(in_path, mask_args)
        except:
            mask_args.set_mask_color_random(use_texture=False)
            masked_image, mask, mask_binary_array, original_image = mask_image(in_path, mask_args)

        if not os.path.exists(os.path.dirname(out_paths[i])):
            os.makedirs(os.path.dirname(out_paths[i]))

        for j in range(len(mask)):
            img = masked_image[j]
            cv2.imwrite(out_paths[i], img)

    pass


if __name__ == '__main__':
    main()
