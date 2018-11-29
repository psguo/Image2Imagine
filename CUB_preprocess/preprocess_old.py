import numpy as np
import pickle

data_dir = './Data/'

def read_file(name):
    with open(data_dir+name) as f:
        for line in f:
            yield line

def parse_save():
  # 1. Parse attributes
  attr_dict = {}
  cat_dict = {}
  cat_list = []
  for attr in read_file('CUB_200_2011/attributes/attributes.txt'):
      if not attr:
          continue
      attribute_id = int(attr.split(' ',1)[0].strip())
      name = attr.split(' ',1)[1].strip()
      cat = name.split('::')[0]
      val = name.split('::')[1]
      if cat not in cat_dict:
          cat_dict[cat] = []
          cat_list.append(cat)

      cat_id = cat_list.index(cat)
      val_id = len(cat_dict[cat])
      cat_dict[cat].append(val)
      attr_dict[attribute_id] = [cat_id, val_id]

  # print(cat_list)
  # print(cat_dict)

  assert (len(cat_list) == len(set(cat_list)))

  # print(len(cat_list))
  # print(attr_dict)

  max_val_num = 0
  for item in cat_dict.items():
      max_val_num = max(len(item[1]), max_val_num)

  # 2. Parse certainty
  certainty_dict = {}
  certainty_score = {'not visible': 0, 'guessing': 0.2, 'probably': 0.5, 'definitely': 1}
  for cert in read_file('CUB_200_2011/attributes/certainties.txt'):
      if not cert:
          continue
      certainty_dict[int(cert.split(' ',1)[0])] = certainty_score[cert.split(' ',1)[1].strip()]

  # 3. Parse labels
  num_cat = len(cat_list)
  input_data = np.zeros((11788, num_cat, max_val_num)) #[<image_id>, <cat_val> * 25] 3D tensor
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

  labels = np.argmax(input_data, axis=2)

  # 4. Parse dirs
  dirs = {}
  for img_dir in read_file('CUB_200_2011/images.txt'):
    if not img_dir:
      continue
    img_dir_info = img_dir.strip().split(' ')
    dirs[int(img_dir_info[0])-1] = img_dir_info[1]

  save_data = {'data':labels, 'cat_list':cat_list, 'cat_dict':cat_dict, 'dirs': dirs}
  with open('birds.pickle', 'wb') as handle:
    pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def print_vec_str(vec,cat_list,cat_dict):
  for i, val in enumerate(vec):
    print(cat_list[i], ":", cat_dict[cat_list[i]][val])

def test_load(id):
  with open('birds.pickle', 'rb') as handle:
    data = pickle.load(handle)

    print_vec_str(data['data'][id], data['cat_list'], data['cat_dict'])

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread(data_dir+'CUB_200_2011/images/'+data['dirs'][id])
    imgplot = plt.imshow(img)
    plt.show()
    # print(data['dirs'][id])


# parse_save()
test_load(4000)