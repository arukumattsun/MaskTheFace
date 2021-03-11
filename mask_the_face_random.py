import argparse
import glob

import dlib
import face3d
import scipy.io as sio

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

MASK_POS_PTN = {
    'PTN1': [50566, 16194, 59779, 103545, 95063, 88261],
    'PTN2': [50566, 16206, 59779, 103545, 95063, 88261],
    'PTN3': [50869, 16000, 59055, 103545, 95063, 88261],
    'PTN4': [50566, 16194, 59779, 54247, 16308, 2805],
}

POS_PTN = ['PTN1', 'PTN2', 'PTN3', 'PTN4']
POS_PTN_P = [0.4, 0.2, 0.2, 0.2]


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
    parser.add_argument(
        "--use_300wlp",
        action='store_true',
        help="Use 300W-LP facial landmark data",
    )
    return parser.parse_args()


def draw_six_point(pt, img_path):
    image = cv2.imread(img_path)[:, :, ::-1]
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    d.line(pt, width=5, fill="white")
    pil_image.show()


def get_six_point(v, tri):
    ptn = MASK_POS_PTN[np.random.choice(POS_PTN, 1, p=POS_PTN_P)[0]]
    six_pt = np.array([v[tri[ptn[0]][-1], :],
                       v[tri[ptn[1]][-1], :],
                       v[tri[ptn[2]][-1], :],
                       v[tri[ptn[3]][-1], :],
                       v[tri[ptn[4]][-1], :],
                       v[tri[ptn[5]][-1], :]])
    if np.random.rand() > 0.5:
        six_pt[:, 0] = six_pt[:, 0] + (np.random.rand() * 5.0)
        six_pt[:, 1] = six_pt[:, 1] + (np.random.rand() * 5.0)

    return six_pt


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

        self.lamdmarks = None
        self.six_points = None
        self.angle = None

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
        rep_src = args.path
        rep_dst = args.path + '_mask'
    elif is_file:
        image_paths = [args.path]
        file_name, ext = os.path.splitext(os.path.basename(args.path))
        rep_src = file_name
        rep_dst = file_name + '_mask'
    else:
        print("Path is neither a valid file or a valid directory")
        exit()

    path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
    bfm = face3d.morphable_model.MorphabelModel('data/Data/out_jonas/Out/BFM.mat')
    mask_args = MaskArgument(path_to_dlib_model)
    for in_path in tqdm(image_paths):
        mask_args.set_mask_type_random()
        mask_args.set_mask_color_random()

        if args.use_300wlp:
            ext = os.path.basename(in_path).split('.')[-1]
            mat_path = in_path.replace(ext, 'mat')
            mat = sio.loadmat(mat_path)
            shape = mat['Shape_Para'].astype(np.float32)
            exp = mat['Exp_Para'].astype(np.float32)
            pose = mat['Pose_Para'].T.astype(np.float32)
            scale = pose[-1, 0]
            angle = pose[:3, 0]
            trans = pose[3:6, 0]
            yaw = np.rad2deg(angle)[1]
            if np.abs(yaw) > 60:
                continue
            vertices = bfm.generate_vertices(shape, exp)
            v = bfm.transform_3ddfa(vertices, scale, angle, trans)[:, :2]
            v[:, 1] = 450 - v[:, 1] - 1
            six_pt = get_six_point(v, bfm.full_triangles)
            mask_args.lamdmarks = v[bfm.kpt_ind, :]
            mask_args.six_points = six_pt
            mask_args.angle = yaw

            # draw_six_point(six_pt, in_path)
            try:
                masked_image, mask, mask_binary_array, original_image = mask_image_300wlp(in_path, mask_args)
            except:
                mask_args.set_mask_color_random(use_texture=False)
                masked_image, mask, mask_binary_array, original_image = mask_image_300wlp(in_path, mask_args)
        else:
            mat_path = None
            try:
                masked_image, mask, mask_binary_array, original_image = mask_image(in_path, mask_args)
            except:
                mask_args.set_mask_color_random(use_texture=False)
                masked_image, mask, mask_binary_array, original_image = mask_image(in_path, mask_args)

        out_path = in_path.replace(rep_src, rep_dst)
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        if len(mask) != 0:
            cv2.imwrite(out_path, masked_image[0])
            if mat_path is not None:
                shutil.copy2(mat_path, mat_path.replace(args.path, rep_dst))


if __name__ == '__main__':
    main()
