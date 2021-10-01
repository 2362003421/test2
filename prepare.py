import os
import re
import time
import nltk
import re
import string
import pickle
import tensorlayer as tl
from utils import *

dataset = '102flowers'
need_256 = True
cwd = os.getcwd()
img_dir = os.path.join(cwd, '102flowers')
caption_dir = os.path.join(cwd, 'text_c10')
VOC_FIR = cwd + '/vocab.txt'

# 导入标题
caption_sub_dir = tl.files.load_folder_list(caption_dir)
captions_dict = {}
processed_capts = []
for sub_dir in caption_sub_dir:  # 标题列表
    with tl.ops.suppress_stdout():
        files = tl.files.load_file_list(path=sub_dir, regx='^image_[0-9]+\.txt')
        for i, f in enumerate(files):  # i为下标 f为元素
            file_dir = os.path.join(sub_dir, f)
            key = int(re.findall('\d+', f)[0])  # 找到数字分组的第一个数字
            t = open(file_dir, 'r')
            lines = []
            for line in t:
                line = preprocess_caption(line)
                lines.append(line)
                processed_capts.append(tl.nlp.process_sentence(line, start_word="<S>", end_word="</S>"))
            assert len(lines) == 10, "Every flower image have 10 captions"
            captions_dict[key] = lines  # 每个图像都有10个标签
print(" * %d x %d captions found " % (len(captions_dict), len(lines)))

# 创建标题字典
_ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)
vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")

captions_ids = []
try:  # python3
    tmp = captions_dict.items()
except:  # python3
    tmp = captions_dict.iteritems()
for key, value in tmp:
    for v in value:
        captions_ids.append([vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(v)] + [vocab.end_id])
        # add END_ID
captions_ids = np.asarray(captions_ids)
print(" * tokenized %d captions" % len(captions_ids))

# 导入图片
with tl.ops.suppress_stdout():  # 给图片按序号排序
    imgs_title_list = sorted(tl.files.load_file_list(path=img_dir, regx='^image_[0-9]+\.jpg'))
print(" * %d images found, start loading and resizing ..." % len(imgs_title_list))
s = time.time()

images = []
images_256 = []
for name in imgs_title_list:
    img_raw = scipy.misc.imread(os.path.join(img_dir, name))
    img = tl.prepro.imresize(img_raw, size=[64, 64])  # (64, 64, 3)
    img = img.astype(np.float32)
    images.append(img)
    if need_256:  # 如果需要256*256图片
        img = tl.prepro.imresize(img_raw, size=[256, 256])  # (256, 256, 3)
        img = img.astype(np.float32)
        images_256.append(img)

print(" * loading and resizing took %ss" % (time.time() - s))
n_images = len(captions_dict)
n_captions = len(captions_ids)
n_captions_per_image = len(lines)  # 10
print("n_captions: %d n_images: %d n_captions_per_image: %d" % (n_captions, n_images, n_captions_per_image))

captions_ids_train, captions_ids_test = captions_ids[: 8000 * n_captions_per_image], captions_ids[
                                                                                     8000 * n_captions_per_image:]
images_train, images_test = images[:8000], images[8000:]

# 8000个训练数据，剩下189个是测试数据

if need_256:
    images_train_256, images_test_256 = images_256[:8000], images_256[8000:]

n_images_train = len(images_train)
n_images_test = len(images_test)
n_captions_train = len(captions_ids_train)
n_captions_test = len(captions_ids_test)
print("n_images_train:%d n_captions_train:%d" % (n_images_train, n_captions_train))
print("n_images_test:%d  n_captions_test:%d" % (n_images_test, n_captions_test))


def save_all(targets, file):
    with open(file, 'wb') as f:
        pickle.dump(targets, f)


# 保存数据
save_all(vocab, '_vocab.pickle')
save_all((images_train_256, images_train), '_image_train.pickle')
save_all((images_test_256, images_test), '_image_test.pickle')
save_all((n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test), '_n.pickle')
save_all((captions_ids_train, captions_ids_test), '_caption.pickle')
