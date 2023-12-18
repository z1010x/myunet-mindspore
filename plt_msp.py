from fileinput import filename
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator

def loss_plot(name, loss_log_filenames, avg_of=100):
    plt.figure()
    plt.ylim((0, 0.2))

    
    # plt.title('TransUnet loss对比')
    plt.xlabel("step")
    plt.ylabel("loss value")
    filenames = []
    for ind, loss_log_filename in enumerate(loss_log_filenames):
        filename = loss_log_filename.split('/')[-1].split('.')[0]
        filenames.append(filename)
        num = avg_of[ind]
        # x = [i for i in range(num)]
        loss = []
        ind = 0
        avg_loss = 0
        with open(loss_log_filename, 'r') as f:        
            lines = f.readlines()
            for line in lines: # epoch: 5 step: 1, loss is 0.009454826824367046
                if 'loss:[' in line:
                    bl = float(line.split('loss:[')[-1].split('/')[0])
                    if ind < num:
                        ind += 1
                        avg_loss += bl
                    else:
                        loss.append(avg_loss / ind)
                        avg_loss = 0
                        ind = 0
                elif 'loss is ' in line:
                    bl = float(line.split('loss is ')[-1])
                    if ind < num:
                        ind += 1
                        avg_loss += bl
                    else:
                        loss.append(avg_loss / ind)
                        avg_loss = 0
                        ind = 0
                elif 'train_loss:' in line:
                    bl = float(line.split('train_loss:')[-1].strip())
                    if ind < num:
                        ind += 1
                        avg_loss += bl
                    else:
                        loss.append(avg_loss / ind)
                        avg_loss = 0
                        ind = 0

        plot_save_path = r'plot/'
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)
        
        x = range(len(loss))
        plt.xlim(right=230)
        plt.plot(x,loss,label='loss')
    save_loss = plot_save_path + name + '_loss.jpg'
    # plt.legend(labels=['TransUnet SGD Loss','TransUnet Adam Loss'])
    filenames = ['U-Net', 'ResUNet-34','Attention UNet','TransUNet']
    plt.legend(filenames)
    plt.savefig(save_loss, dpi=400)

## 画accuacy
def acc_plot(name, loss_log_filenames):
    
    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim((0.6, 1.0))
    # plt.title('TransUnet loss对比')
    plt.xlabel("epoch")
    plt.ylabel("dice_coeff value")
    filenames = []
    for ind, loss_log_filename in enumerate(loss_log_filenames):
        filename = loss_log_filename.split('/')[-1].split('.')[0]
        filenames.append(filename)
        acc = []
        with open(loss_log_filename, 'r') as f:  
            lines = f.readlines()
            for line in lines: 
                if 'Accuracy:  ' in line:
                    a = float(line.split('Accuracy:  ')[-1])
                    acc.append(a)

        plot_save_path = r'plot_infos/'
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)
        x = list(range(len(acc)))
        plt.plot(x,acc,label='accuracy')
    # plt.xlim(0, 20)
    save_acc = plot_save_path + name + '_accuracy.jpg'
    # l = ['TransUnet test accuracy','Unet test accuracy',]
    l = filenames
    plt.legend(labels=l)
    plt.savefig(save_acc, dpi=400)

if __name__ == "__main__":
    name = 'unet'
    filenames = ['./log/train_unet_422_lr-5.log']
    loss_plot(name, filenames, avg_of=[100,400,200,200])
    acc_plot(name, filenames)