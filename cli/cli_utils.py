import matplotlib.pyplot as plt


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


strawb = """
            _
        /> //  __
    ___/ \// _/ /
  ,' , \_/ \/ _/__
 /    _/ |--\  `  `~,
' , ,/  /`\ / `  `   `,
|    |  |  \> `  `  ` |
|  ,  \/ ' '    `  `  /
`,   '  '    ' `  '  /
  \ `      '  ' ,  ,'
   \ ` ` '    ,  ,/
    `,  `  '  , ,'
      \ `  ,   /
       `~----~'"""



def plot_multiple_img(imgs, gray=False, titles=''):
    """
    Plots multiple images in a single figure.

    Args:
        imgs (list): list of images to plot
        gray (boolean): if True, plots in grayscale
        titles (list): list of titles for each image
    """
    cnt = len(imgs)
    _, axs = plt.subplots(1,len(imgs), figsize=(18, 10))
    for i in range(cnt):
      if gray:
        axs[i].imshow(imgs[i], interpolation='nearest', cmap='gray')
      else:
        axs[i].imshow(imgs[i], interpolation='nearest')
      if titles:
        axs[i].set_title(titles[i])
      axs[i].set_xticks([])
      axs[i].set_yticks([])
    plt.show()
