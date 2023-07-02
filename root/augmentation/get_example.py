# import  matplotlib as plt
#
# def example():
#     n_examples = 9
#
#     fig, ax = plt.subplots(nrows=n_examples, ncols=(len(augmenters) + 1), figsize=(10, 10))
#
#     for i in range(n_examples):
#         random_symbols = int(np.random.uniform(0, len(train_val_files)))
#         img_orig = load_image(train_val_files[random_symbols])
#         img_label = train_val_files[random_symbols].parent.name
#
#         img_label = " ".join(map(lambda x: x.capitalize(), \
#                                  img_label.split('_')))
#         plt.subplots_adjust(left=0, bottom=0, right=1,
#                             top=n_examples // 3 * 0.4, wspace=0.3 * ((len(augmenters) + 1) // 3), hspace=0)
#         # wspace горизонтальные отступы(слева и справа)
#         # для 3ёх изображений plt.subplots_adjust(left=0, bottom=0, right=1, top=0.6, wspace=0.3, hspace=0)
#         # hspace=(len(augmenters)+1)//3*0.3) n_examples//3*0.5
#         ax[i][0].imshow(img_orig, cmap='gray')
#         ax[i][0].set_title(img_label)
#         # ax[i][0].axis('off')
#
#         for j, (augmenter_name, augmenter) in enumerate(augmenters.items()):
#             img_aug = augmenter(img_orig)
#             ax[i][j + 1].imshow(img_aug, cmap='gray')
#             ax[i][j + 1].set_title(augmenter_name)
#             # ax[i][j + 1].axis('off')