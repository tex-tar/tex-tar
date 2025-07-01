import torch
def labelize(image_file_name,label_split,total):
    get_list = image_file_name.split('/')[-1].split('_')[0].split('-')
    get_list[-1] = get_list[-1].split('.')[0]
    
    label = []
    flag=0
    pos=0
    for i in range(len(label_split)):
        if label_split[i]==-2:
            pos+=1
        elif label_split[i]==-1:
            pos+=1
            label.append(torch.zeros(total[i]))
        elif label_split[i]==0:
          
            continue  
        else:
            if get_list[pos][0]=='[':

                numbers = get_list[pos].strip("[]").split(",")

                numbers = [int(float(num)//16) for num in numbers]  
                # print(numbers)
                label.append(torch.tensor(numbers))        
              
            else:
                # classification
                one_hot = torch.zeros(label_split[i])
                one_hot[int(get_list[pos])]=1.0
                label.append(one_hot)
                pos+=1
                
    # assert pos==1
    to_ret = torch.cat(label)
    return to_ret