import numpy as np
import pickle

data_dir = './Data/'

wanted_cats = ['has_bill_shape', 'has_wing_color', 'has_shape', 'has_back_pattern', 'has_primary_color']
wanted_cats_num = [0] * len(wanted_cats)
NUM_IMAGES = 11788

def read_file(name):
    with open(data_dir+name) as f:
        for line in f:
            yield line

def parse_save():
  # 1. Parse attributes
  attr_dict = {}
  cat_dict = {}
  cat_list = []
  wanted_attr_id_dict = {}
  wanted_attr_id_reverse_dict = {}
  attr_mapping = {}
  count = 0
  for attr in read_file('CUB_200_2011/attributes/attributes.txt'):
      if not attr:
          continue
      attribute_id = int(attr.split(' ',1)[0].strip())
      name = attr.split(' ',1)[1].strip()
      attr_mapping[attribute_id] = name
      cat = name.split('::')[0]
      val = name.split('::')[1]
      if cat not in cat_dict:
          cat_dict[cat] = []
          cat_list.append(cat)

      cat_id = cat_list.index(cat)
      val_id = len(cat_dict[cat])
      cat_dict[cat].append(val)
      attr_dict[attribute_id] = [cat_id, val_id]

      if cat in wanted_cats:
        wanted_attr_id_dict[attribute_id] = count
        wanted_attr_id_reverse_dict[count] = attribute_id
        count += 1
        wanted_cats_num[wanted_cats.index(cat)] += 1

  # print(cat_list)
  # print(cat_dict)

  assert (len(cat_list) == len(set(cat_list)))

  # wanted_cats_id = []
  # wanted_attr_id_dict = {}
  # count = 0
  # for cat in wanted_cats:
  #   cat_id = cat_list.index(cat)
  #   wanted_cats_id.append(cat_id)
  #   if cat_id not in wanted_attr_id_dict:
  #     wanted_attr_id_dict[cat_id] = {}
  #   wanted_attr_id_dict[cat_id][] =

  # print(len(cat_list))
  # print(attr_dict)

  max_val_num = 0
  for item in cat_dict.items():
      max_val_num = max(len(item[1]), max_val_num)

  # 2. Parse certainty
  certainty_dict = {}
  # certainty_score = {'not visible': 0, 'guessing': 0.2, 'probably': 0.5, 'definitely': 1}
  certainty_score = {'not visible': 0, 'guessing': 1, 'probably': 1, 'definitely': 1}
  for cert in read_file('CUB_200_2011/attributes/certainties.txt'):
      if not cert:
          continue
      certainty_dict[int(cert.split(' ',1)[0])] = certainty_score[cert.split(' ',1)[1].strip()]

  # 3. Parse labels
  num_cat = len(cat_list)
  input_data = np.zeros((NUM_IMAGES, num_cat, max_val_num)) #[<image_id>, <cat_val> * 25] 3D tensor
  input_data_new = np.zeros((NUM_IMAGES, len(wanted_attr_id_dict)))
  for label in read_file('CUB_200_2011/attributes/image_attribute_labels.txt'):
      if not label:
          continue
      label_items = label.strip().split(' ') #[<image_id>, <attribute_id>, <is_present>, <certainty_id>, <worker_id>]
      image_id = int(label_items[0])-1
      attr_id = int(label_items[1])
      cat_id = attr_dict[attr_id][0]
      val_id = attr_dict[attr_id][1]

      is_present = int(label_items[2])
      certainty_id = int(label_items[3])

      # print(image_id)
      # print(cat_id)
      # print(val_id)
      # print("---------")
      input_data[image_id][cat_id][val_id] += is_present * certainty_dict[certainty_id]

      if attr_id in wanted_attr_id_dict:
        input_data_new[image_id][wanted_attr_id_dict[attr_id]] += is_present * certainty_dict[certainty_id]

  labels = np.argmax(input_data, axis=2)

  # 4. Parse dirs
  dirs = {}
  for img_dir in read_file('CUB_200_2011/images.txt'):
    if not img_dir:
      continue
    img_dir_info = img_dir.strip().split(' ')
    dirs[int(img_dir_info[0])-1] = img_dir_info[1]


  # 5. New random attr
  NUM_RANDOM_TEXT_PER_IMAGE = 5
  random_texts = np.expand_dims(input_data_new, axis=1)
  random_texts = np.repeat(random_texts, NUM_RANDOM_TEXT_PER_IMAGE, axis=1)

  import random
  def create_random_one_hot(dim):
    rand_one_hot = np.zeros(dim)
    rand_one_hot[random.randint(0,dim-1)] = 1.0
    return rand_one_hot

  for i in range(NUM_IMAGES):
    for j in range(NUM_RANDOM_TEXT_PER_IMAGE):
      rand_cat_id = random.randint(0,len(wanted_cats_num)-1)
      start = sum(wanted_cats_num[:rand_cat_id])
      end = start + wanted_cats_num[rand_cat_id]
      random_texts[i][j][start:end] = create_random_one_hot(wanted_cats_num[rand_cat_id])

  # new_random_attr = []
  # import random
  # for cat_num in wanted_cats_num:
  #   rand_attr = [0] * cat_num
  #   rand_attr[random.randint(0,cat_num-1)] = 1.0
  #   new_random_attr += rand_attr

  # 6. Save data
  # save_data = {'data':labels, 'cat_list':cat_list, 'cat_dict':cat_dict, 'dirs': dirs}
  save_data = {'data':input_data_new, 'attr_mapping':attr_mapping, 'dirs': dirs,
               'wanted_attr_id_reverse_dict':wanted_attr_id_reverse_dict, 'wanted_cats_num':wanted_cats_num,
               'random_texts': random_texts}
  with open('birds.pickle', 'wb') as handle:
    pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def print_vec_str(vec,attr_mapping, wanted_attr_id_reverse_dict):
  for i, val in enumerate(vec):
    attr_id = wanted_attr_id_reverse_dict[i]
    print(attr_mapping[attr_id], ":", val)

def test_load(id):
  with open('birds.pickle', 'rb') as handle:
    data = pickle.load(handle)

    print_vec_str(data['data'][id], data['attr_mapping'], data['wanted_attr_id_reverse_dict'])

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread(data_dir+'CUB_200_2011/images/'+data['dirs'][id])
    imgplot = plt.imshow(img)
    plt.show()
    # print(data['dirs'][id])


# parse_save()
test_load(4000)