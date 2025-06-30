import numpy as np
import cv2
import math


import random
import numpy as np  
np.random.seed(42)
random.seed(42)

def aspect_conservative_resize(orignal_image,height=170,width=1200):
    w = int(width)
    h = int(orignal_image.shape[0]*(width/orignal_image.shape[1]))
    if h>height:
        w=int(height*orignal_image.shape[1]/orignal_image.shape[0])
        h = height
    return cv2.resize(orignal_image,(w,h),interpolation=cv2.INTER_LINEAR)

def aspect_conservative_resize_height(orignal_image,height=170,width=1200):
    h = int(height)
    w = int(orignal_image.shape[1]*(height/orignal_image.shape[0]))
    if w>width:
        h=int(width*orignal_image.shape[0]/orignal_image.shape[1])
        w = width
    return cv2.resize(orignal_image,(w,h),interpolation=cv2.INTER_LINEAR)

def centralizer(orignal_image,height=170,width=1200):
    pre_processed_image = orignal_image
    if orignal_image.shape[1]>width:
        pre_processed_image = aspect_conservative_resize(orignal_image,height,width)
    
    elif orignal_image.shape[0]>height:
        pre_processed_image = aspect_conservative_resize_height(orignal_image,height,width)
    
    plain_image = np.zeros((height,width,3),dtype=np.float32)
    plain_image.fill(255)
    width_centering_factor = (plain_image.shape[1] - pre_processed_image.shape[1])//2
    height_centering_factor = (plain_image.shape[0] - pre_processed_image.shape[0])//2  
    plain_image[height_centering_factor:pre_processed_image.shape[0]+height_centering_factor,width_centering_factor:pre_processed_image.shape[1]+width_centering_factor] = pre_processed_image[:,:]

    return plain_image


def findCentroidAndReassignCoordinates(bbox_array,h,w,is_pad):
    dimen = bbox_array
    centroid = np.array([0,0,0,0])
    for dim in dimen:
        centroid[0]+=dim[0]
        centroid[1]+=dim[1]
        centroid[2]+=dim[2]
        if not is_pad:
            centroid[3]+=dim[3]
        else:
            centroid[3]+=dim[3]+math.ceil(0.10*abs(dim[3]-dim[1]))
    centroid = centroid/len(dimen)
    centroid_toret = [(centroid[1]+centroid[3])/(2*h) , (centroid[0]+centroid[2])/(2*w)]
    # centroid finalized
    reassigned_array = []
    for dim in dimen:
        coord = (dim - centroid)
        reassigned_array.append([(coord[1]+coord[3])/(2*h),(coord[0]+coord[2])/(2*w)])
    return reassigned_array,centroid_toret

def get_crops_after_expansion(new_bb,counter,dimensions_set,original_dim,args):
    pointer_2 = 0
    crops = []
    cnt=0

    if args.type=='word':
        # normalized_centralized_bb = centralizer(new_bb,height=56,width=400)
        crops.append(new_bb)
        dimensions_set.append(original_dim)
    else:
        while pointer_2+math.floor((args.aspect_ratio)*new_bb.shape[0])<new_bb.shape[1]:
            if cnt+counter<args.seq_size:
                crops.append(new_bb[:,pointer_2 : pointer_2+math.floor((args.aspect_ratio)*new_bb.shape[0])])
                cnt+=1
                dimensions_set.append([original_dim[0]+pointer_2,original_dim[1],original_dim[0]+pointer_2+math.floor((args.aspect_ratio)*new_bb.shape[0]),original_dim[3]])
            else:
                return crops
            pointer_2+=math.floor((args.aspect_ratio)*new_bb.shape[0])
            
        if cnt+counter<args.seq_size:
            if pointer_2-new_bb.shape[1]<-math.floor((args.aspect_ratio)*new_bb.shape[0])//5:
                crops.append(new_bb[:,new_bb.shape[1]-math.floor((args.aspect_ratio)*new_bb.shape[0]) : new_bb.shape[1]])
                dimensions_set.append([original_dim[0]+new_bb.shape[1]-math.floor((args.aspect_ratio)*new_bb.shape[0]),original_dim[1],original_dim[0]+new_bb.shape[1],original_dim[3]])
                cnt+=1
    # print(crops)
    return crops

def n_get_blocks(word_bb,counter,dimensions_set,original_dim,args):

    h = word_bb.shape[0]
    w = word_bb.shape[1]

    if args.type=='word':
        return get_crops_after_expansion(word_bb,counter,dimensions_set,original_dim=original_dim,args=args)
    else:
        if w/h >= args.aspect_ratio : 
            return get_crops_after_expansion(word_bb,counter,dimensions_set,original_dim=original_dim,args=args)
            # seperate into blocks
        else:
            # expand the image
            desired_width = math.ceil((args.aspect_ratio*h))
            expand_bb = np.zeros((h,desired_width,3))
            # repeat the image
            curr_width=0
            while curr_width < desired_width:
                expand_bb[:,curr_width : min(w+curr_width,desired_width)] = word_bb[:,:min(w,desired_width-curr_width)]
                curr_width += w

            expand_bb = expand_bb.astype(np.uint8)
        return get_crops_after_expansion(expand_bb,counter,dimensions_set,original_dim=original_dim,args=args)
    
def distance_metric(data,a,b):
    weighted_y = np.random.randint(1, 11)
    weighted_x = np.random.randint(3*weighted_y,3*11)


    wx = weighted_x*np.abs(data[:,0]-a)
    wy = weighted_y*np.abs(data[:,1]-b)

    weighted_arr = np.concatenate([wx[:,np.newaxis],wy[:,np.newaxis]],axis=1)
    distances = np.max(weighted_arr,axis=1)
    return distances


def nearest_points_from_point(data,bbox_array,image, a, b,args):
    if len(data) == 0:
        raise ValueError("Input array is empty.")

    distances = distance_metric(data,a,b)

    # Get the indices of the nearest points
    nearest_order = np.argsort(distances)

    counter=0
    iterations = 0
    final_cropped = {}
    dimensions_full_set = []
    while counter < args.seq_size and iterations < len(nearest_order):
        dimensions_set = []
        dim = bbox_array[nearest_order[iterations]]
        """
        ADD PADDING at the bottom to cater to underline
        """
        if args.pad=='n':
            extended_dim3 = dim[3]
        else:
            extended_dim3 = dim[3]+math.ceil(0.10*abs(dim[3]-dim[1]))
        crops= n_get_blocks(image[dim[1]:extended_dim3,dim[0]:dim[2]],counter,dimensions_set,original_dim=[dim[0],dim[1],dim[2],extended_dim3],args=args)
        dimensions_full_set.extend(dimensions_set)
        # print(len(dimensions_set),"dim set")
       

        counter += len(crops)
        final_cropped[nearest_order[iterations]] = crops
        iterations+=1
    
    if len(dimensions_full_set)!=args.seq_size:
            return -1
    
    return final_cropped,dimensions_full_set,counter


def crop_frame_around_points(choice_set,data,bbox_array,image,args):
    ind_rand = np.random.choice(choice_set[choice_set!=-1])
    anchor_x,anchor_y = data[ind_rand]
    
    try:
        cropped_data,dimensions_full_set,cntr = nearest_points_from_point(data,bbox_array,image,anchor_x,anchor_y,args)
        keys = list(cropped_data.keys())
        if cntr<args.seq_size:
            raise("Number of patches in a crop<seq_size!")
        return keys,data[keys],cropped_data,dimensions_full_set
    except Exception as e:
        # print(e)
        return -1
    
# def extract_words(page_image,image_path,bbjson,params,extract_dict):
#     key = image_path.split('/')[-1]
#     image_path = image_path.split('/')[-1]
#     # prmath.ceil(extract_dict["offset"])
#     for bbid in range(len(bbjson[key])):
#         dim = bbjson[key][bbid]["bb_dim"]
#         for k in extract_dict:
#             if(k=="T1"):
#                 if bbjson[key][bbid]["bb_ids"][0]["attb"]["b+i"]==True:
#                     cv2.imwrite(f'{extract_dict["T1"]}/3_{image_path}_{bbid}.png',page_image[dim[1]:dim[3] + math.ceil(params["offset"]*abs(dim[3]-dim[1])),dim[0]:dim[2]])
#                 if bbjson[key][bbid]["bb_ids"][0]["attb"]["bold"]==True:
#                     cv2.imwrite(f'{extract_dict["T1"]}/1_{image_path}_{bbid}.png',page_image[dim[1]:dim[3] + math.ceil(extract_dict["offset"]*abs(dim[3]-dim[1])),dim[0]:dim[2]])
#                 if bbjson[key][bbid]["bb_ids"][0]["attb"]["italic"]==True:
#                     cv2.imwrite(f'{extract_dict["T1"]}/2_{image_path}_{bbid}.png',page_image[dim[1]:dim[3] + math.ceil(extract_dict["offset"]*abs(dim[3]-dim[1])),dim[0]:dim[2]])
#                 if bbjson[key][bbid]["bb_ids"][0]["attb"]["no_bi"]==True:
#                     cv2.imwrite(f'{extract_dict["T1"]}/0_{image_path}_{bbid}.png',page_image[dim[1]:dim[3] + math.ceil(extract_dict["offset"]*abs(dim[3]-dim[1])),dim[0]:dim[2]])

#             if(k=="T2"):
#                 if bbjson[key][bbid]["bb_ids"][0]["attb"]["u+s"]==True:
#                     cv2.imwrite(f'{extract_dict["T2"]}/3_{image_path}_{bbid}.png',page_image[dim[1]:dim[3] + math.ceil(extract_dict["offset"]*abs(dim[3]-dim[1])),dim[0]:dim[2]])
#                 if bbjson[key][bbid]["bb_ids"][0]["attb"]["underlined"]==True:
#                     cv2.imwrite(f'{extract_dict["T2"]}/1_{image_path}_{bbid}.png',page_image[dim[1]:dim[3] + math.ceil(extract_dict["offset"]*abs(dim[3]-dim[1])),dim[0]:dim[2]])
#                 if bbjson[key][bbid]["bb_ids"][0]["attb"]["strikeout"]==True:
#                     cv2.imwrite(f'{extract_dict["T2"]}/2_{image_path}_{bbid}.png',page_image[dim[1]:dim[3] + math.ceil(extract_dict["offset"]*abs(dim[3]-dim[1])),dim[0]:dim[2]])
#                 if bbjson[key][bbid]["bb_ids"][0]["attb"]["no_us"]==True:
#                     cv2.imwrite(f'{extract_dict["T2"]}/0_{image_path}_{bbid}.png',page_image[dim[1]:dim[3] + math.ceil(extract_dict["offset"]*abs(dim[3]-dim[1])),dim[0]:dim[2]])
                    
def extract_cw(page_image,image_path,bbjson,cw_json,extract_dict,params,args,cw_offset):
    key = image_path.split('/')[-1]
    image_path = image_path.split('/')[-1]
    
    # print(image_path)
    context_window_offset = cw_offset
    prev_id = context_window_offset-1
    for cwi in range(len(cw_json[key])):
        # print(cwi)
        context_window = cw_json[key][cwi]
        # assert len(context_window)==100
        
        # print(len(context_window))
        cwid = cwi+context_window_offset
        error=0
        
        # print(cwid,prev_id)
        assert cwid == prev_id+1
        
        num_writes=0
        num_crops_in_a_window = 0
        for idx,bbid in enumerate(context_window):
            dim = bbjson[key][int(bbid)]["bb_dim"]
            dimensions_set = []
            """
            ADD PADDING at the bottom to cater to underline
            """
            if args.pad=='n':
                extended_dim3 = dim[3]
            else:
                extended_dim3 = dim[3]+math.ceil(params["offset"]*abs(dim[3]-dim[1]))
            crops= n_get_blocks(page_image[dim[1]:extended_dim3,dim[0]:dim[2]],-1,dimensions_set,original_dim=[dim[0],dim[1],dim[2],extended_dim3],args=args)
            # print(len(crops))
            try:
                #print(len(crops),context_windo[bbid])
                assert len(crops)==context_window[bbid]
                num_crops_in_a_window+=len(crops)
            except:
                #print(len(crops),context_window[bbid])
                crops = crops[:context_window[bbid]]
                num_crops_in_a_window+=len(crops)
                error+=1
                
            bbid = int(bbid)
            for ind,crop in enumerate(crops):
                build_name = ""
                for k in extract_dict:
                    if(k=="T1" and extract_dict[k]==True):
                        # print("hello")
                        #print(bbjson[key][bbid])
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["b+i"]==True:
                            build_name+="3-"
                            # cv2.imwrite(f'{extract_dict["T1"]}/3_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["bold"]==True:
                            build_name+="1-"
                            # cv2.imwrite(f'{extract_dict["T1"]}/1_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["italic"]==True:
                            build_name+="2-"
                            # cv2.imwrite(f'{extract_dict["T1"]}/2_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["no_bi"]==True:
                            build_name+="0-"
                            # cv2.imwrite(f'{extract_dict["T1"]}/0_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        # print(build_name)

                    elif(k=="T2" and extract_dict[k]==True):
                        # print("hello")
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["u+s"]==True:
                            build_name+="3-"
                            # cv2.imwrite('{extract_dict["T2"]}/3_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["underlined"]==True:
                            build_name+="1-"
                            # cv2.imwrite(f'{extract_dict["T2"]}/1_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["strikeout"]==True:
                            build_name+="2-"
                            # cv2.imwrite(f'{extract_dict["T2"]}/2_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["no_us"]==True:
                            build_name+="0-"
                            # cv2.imwrite(f'{extract_dict["T2"]}/0_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                    elif(k=="T3" and extract_dict[k]== True):
                        build_name+= str(bbjson[key][bbid]["bb_ids"][0]["attb"]['textcolor_hisam_gmm'])+"-"
                        '''if bbjson[key][bbid]["bb_ids"][0]["attb"]["redc"]==True:
                            build_name+="0-"
                            # cv2.imwrite('{extract_dict["T2"]}/3_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["cyanc"]==True:
                            build_name+="1-"
                            # cv2.imwrite(f'{extract_dict["T2"]}/1_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["greenc"]==True:
                            build_name+="2-"
                            # cv2.imwrite(f'{extract_dict["T2"]}/2_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["bluec"]==True:
                            build_name+="3-"
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["yellow"]==True:
                            build_name+="4-"
                            # cv2.imwrite('{extract_dict["T2"]}/3_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["magnetac"]==True:
                            build_name+="5-"
                            # cv2.imwrite(f'{extract_dict["T2"]}/1_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["blackc"]==True:
                            build_name+="6-"
                            # cv2.imwrite(f'{extract_dict["T2"]}/2_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["whitec"]==True:
                            build_name+="7-"
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["greyc"]==True:
                            build_name+="8-"
                            # cv2.imwrite(f'{extract_dict["T2"]}/2_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["brownc"]==True:
                            build_name+="9-" '''

                    elif(k=="T4" and extract_dict[k]== True):
                        build_name+= str(bbjson[key][bbid]["bb_ids"][0]["attb"]['highlight_hisam'])+"-"
                        '''if bbjson[key][bbid]["bb_ids"][0]["attb"]["redh"]==True:
                            build_name+="0-"
                            # cv2.imwrite('{extract_dict["T2"]}/3_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["cyanh"]==True:
                            build_name+="1-"
                            # cv2.imwrite(f'{extract_dict["T2"]}/1_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["greenh"]==True:
                            build_name+="2-"
                            # cv2.imwrite(f'{extract_dict["T2"]}/2_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["blueh"]==True:
                            build_name+="3-"
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["yellowh"]==True:
                            build_name+="4-"
                            # cv2.imwrite('{extract_dict["T2"]}/3_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["magnetah"]==True:
                            build_name+="5-"
                            # cv2.imwrite(f'{extract_dict["T2"]}/1_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["blackh"]==True:
                            build_name+="6-"
                            # cv2.imwrite(f'{extract_dict["T2"]}/2_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["whiteh"]==True:
                            build_name+="7-"
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["greyh"]==True:
                            build_name+="8-"
                            # cv2.imwrite(f'{extract_dict["T2"]}/2_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
                        if bbjson[key][bbid]["bb_ids"][0]["attb"]["brownh"]==True:
                            build_name+="9-" '''     

                    else:
                        continue
                    # print(build_name)
                #assert len(build_name)==4
                num_writes+=1
                cv2.imwrite(f'{params["path"]}/{build_name[:-1]}_{image_path}_{bbid}_CW_{cwid}_{ind}.png',crop)
        prev_id=cwid
        #print(error)
        assert error<=1
        assert num_crops_in_a_window==125
        
        print("Num writes",num_writes)
        # return -1
        
    context_window_offset+=len(cw_json[key])
    
    return context_window_offset
