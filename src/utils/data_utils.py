import numpy as np
import cv2
import math
import random 
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
    reassigned_array = []
    for dim in dimen:
        coord = (dim - centroid)
        reassigned_array.append([(coord[1]+coord[3])/(2*h),(coord[0]+coord[2])/(2*w)])
    return reassigned_array,centroid_toret

def get_crops_after_expansion(new_bb, counter, dimensions_set, original_dim,
                              context_window_type: str, aspect_ratio: float, seq_size: int):
    pointer_2 = 0
    crops = []
    cnt=0

    if context_window_type == 'word CW': 
        crops.append(new_bb)
        dimensions_set.append(original_dim)
    else:
        patch_width = math.floor(aspect_ratio * new_bb.shape[0])
        
        if patch_width == 0:
            print(f"  ERROR (get_crops_after_expansion): Calculated patch_width is 0 for H={new_bb.shape[0]}, AR={aspect_ratio}. Returning empty crops.", file=sys.stderr)
            return []

        while pointer_2 + patch_width < new_bb.shape[1]:
            if cnt + counter < seq_size: 
                crops.append(new_bb[:,pointer_2 : pointer_2 + patch_width])
                cnt+=1
                dimensions_set.append([original_dim[0]+pointer_2,original_dim[1],original_dim[0]+pointer_2+patch_width,original_dim[3]])
            else:
                return crops
            pointer_2 += patch_width
            
        if cnt+counter < seq_size: 
            if pointer_2-new_bb.shape[1] < -patch_width // 5: 
                crops.append(new_bb[:,new_bb.shape[1]-patch_width : new_bb.shape[1]])
                dimensions_set.append([original_dim[0]+new_bb.shape[1]-patch_width,original_dim[1],original_dim[0]+new_bb.shape[1],original_dim[3]])
                cnt+=1
    return crops

def n_get_blocks(word_bb, counter, dimensions_set, original_dim,
                 context_window_type: str, aspect_ratio: float, seq_size: int):

    h = word_bb.shape[0]
    w = word_bb.shape[1]


    if context_window_type == 'word CW': 
        return get_crops_after_expansion(word_bb,counter,dimensions_set,original_dim=original_dim,
                                         context_window_type=context_window_type, aspect_ratio=aspect_ratio, seq_size=seq_size) 
    else: 
        if w/h >= aspect_ratio: 
            return get_crops_after_expansion(word_bb,counter,dimensions_set,original_dim=original_dim,
                                             context_window_type=context_window_type, aspect_ratio=aspect_ratio, seq_size=seq_size) 
        else:
            desired_width = math.ceil((aspect_ratio*h)) 
            if desired_width == 0:
                print(f"  ERROR (n_get_blocks): Desired width is 0 for H={h}, AR={aspect_ratio}. Cannot expand image.", file=sys.stderr)
                return [] 
            
            expand_bb = np.zeros((h,desired_width,3), dtype=np.uint8) 
            curr_width=0
            while curr_width < desired_width:
                copy_width = min(w, desired_width - curr_width)
                if copy_width <= 0: 
                    break
                expand_bb[:,curr_width : curr_width + copy_width] = word_bb[:, :copy_width]
                curr_width += w 

            return get_crops_after_expansion(expand_bb,counter,dimensions_set,original_dim=original_dim,
                                             context_window_type=context_window_type, aspect_ratio=aspect_ratio, seq_size=seq_size) 
def distance_metric(data,a,b):
    weighted_y = 5 
    weighted_x = 10 

    wx = weighted_x*np.abs(data[:,0]-a)
    wy = weighted_y*np.abs(data[:,1]-b)

    weighted_arr = np.concatenate([wx[:,np.newaxis],wy[:,np.newaxis]],axis=1)
    distances = np.max(weighted_arr,axis=1)
    return distances

def nearest_points_from_point(data,bbox_array,image, a, b,
                              seq_size: int, pad_cropped_frame: str, aspect_ratio: float, context_window_type: str):
    if len(data) == 0:
        raise ValueError("Input array of bbox_centres is empty.")

    distances = distance_metric(data,a,b)

    nearest_order = np.argsort(distances)

    counter=0
    iterations = 0
    final_cropped = {}
    dimensions_full_set = []

    while counter < seq_size and iterations < len(nearest_order): 
        dimensions_set = [] 
        dim = bbox_array[nearest_order[iterations]]

        if dim[2] <= dim[0] or dim[3] <= dim[1]:
            print(f"  WARNING: Invalid bbox dimension {dim} for bbox_array[{nearest_order[iterations]}]. Skipping.", file=sys.stderr)
            iterations += 1
            continue 
        extended_dim3 = dim[3]
        if pad_cropped_frame == 'y': 
            extended_dim3 = dim[3]+math.ceil(0.10*abs(dim[3]-dim[1]))
        
        y1_slice = max(0, dim[1])
        y2_slice = min(image.shape[0], extended_dim3)
        x1_slice = max(0, dim[0])
        x2_slice = min(image.shape[1], dim[2])

        word_bb_image = image[y1_slice:y2_slice, x1_slice:x2_slice]

        if word_bb_image.size == 0 or word_bb_image.shape[0] == 0 or word_bb_image.shape[1] == 0:
            print(f"  WARNING: Extracted word_bb_image is empty/invalid for bbox {dim}. Skipping crops generation for this bbox.", file=sys.stderr)
            iterations += 1
            continue 
        crops = n_get_blocks(
            word_bb_image, 
            counter,       
            dimensions_set,
            original_dim=[dim[0],dim[1],dim[2],extended_dim3], 
            context_window_type=context_window_type, 
            aspect_ratio=aspect_ratio,          
            seq_size=seq_size                    
        )
        
        if not crops: 
            print(f"  WARNING: n_get_blocks returned no crops for bbox {dim}. Skipping this bbox.", file=sys.stderr)
            iterations += 1
            continue


        dimensions_full_set.extend(dimensions_set)
       
        counter += len(crops)
        final_cropped[nearest_order[iterations]] = crops
        iterations+=1
    
    if len(dimensions_full_set) != seq_size: 
            raise ValueError(f"Could not collect at least {seq_size} patches/words for a context window. Found {len(dimensions_full_set)}.")
    
    return nearest_order[0:iterations], final_cropped, dimensions_full_set, counter

def crop_frame_around_points(choice_set, data, bbox_array, image,
                             seq_size: int, pad: str, aspect_ratio: float, context_window_type: str):
    
    available_choices = choice_set[choice_set != -1]
    if len(available_choices) == 0:
        raise ValueError("No unvisited bounding boxes available to choose a starting point from.")

    ind_rand = np.random.choice(available_choices)
    anchor_x,anchor_y = data[ind_rand]
    
    try:
        visited_indices_in_sequence, cropped_data, dimensions_full_set, cntr = nearest_points_from_point(
            data=data,
            bbox_array=bbox_array,
            image=image,
            a=anchor_x,
            b=anchor_y,
            seq_size=seq_size,
            pad_cropped_frame=pad, 
            aspect_ratio=aspect_ratio,
            context_window_type=context_window_type
        )
        if cntr != seq_size:
            raise ValueError(f"nearest_points_from_point did not return exactly {seq_size} crops.")

        return visited_indices_in_sequence, data[visited_indices_in_sequence], cropped_data, dimensions_full_set
    except Exception as e:
        raise RuntimeError(f"Could not make a valid context window: {e}") from e

def extract_cw(page_image, image_path, bbjson, cw_json, extract_dict, params,
               pad_cropped_frame: str, context_window_type: str, sequence_size: int, aspect_ratio: float,
               cw_offset):
    key = image_path.split('/')[-1]
    image_file_name_only = image_path.split('/')[-1] 

    context_window_offset = cw_offset
    prev_id = context_window_offset-1
    for cwi in range(len(cw_json[key])):
        context_window = cw_json[key][cwi]
        
        cwid = cwi+context_window_offset
        error_count=0
        
        assert cwid == prev_id+1, f"Context window ID mismatch: Expected {prev_id+1}, Got {cwid}"
        
        num_crops_in_a_window_actual = 0
        for bbid_str, expected_num_crops_for_bbox in context_window.items(): 
            bbid = int(bbid_str) 
            
            dim = bbjson[key][bbid]["bb_dim"]
            dimensions_set = []
            
            extended_dim3 = dim[3]
            if pad_cropped_frame == 'y': 
                extended_dim3 = dim[3]+math.ceil(params["offset"]*abs(dim[3]-dim[1]))
            
            y1_slice = max(0, dim[1])
            y2_slice = min(page_image.shape[0], extended_dim3)
            x1_slice = max(0, dim[0])
            x2_slice = min(page_image.shape[1], dim[2])

            word_bb_image = page_image[y1_slice:y2_slice, x1_slice:x2_slice]

            if word_bb_image.size == 0 or word_bb_image.shape[0] == 0 or word_bb_image.shape[1] == 0:
                print(f"  Warning: Empty word_bb_image for {image_file_name_only}, bbid {bbid}. Skipping crops generation for this bbox in extract_cw.", file=sys.stderr)
                continue

            crops = n_get_blocks(
                word_bb_image,
                -1, 
                dimensions_set,
                original_dim=[dim[0],dim[1],dim[2],extended_dim3],
                context_window_type=context_window_type, 
                aspect_ratio=aspect_ratio,             
                seq_size=sequence_size                  
            )
            
            try:
                if len(crops) != expected_num_crops_for_bbox:
                    print(f"  Warning: Mismatch for bbid {bbid} in CW {cwid} ({image_file_name_only}). Expected {expected_num_crops_for_bbox}, Got {len(crops)}. Adjusting list.", file=sys.stderr)
                    crops = crops[:expected_num_crops_for_bbox] 
                    error_count += 1 

                num_crops_in_a_window_actual+=len(crops)
            except Exception as e:
                print(f"Error processing crops for bbid {bbid} in CW {cwid}: {e}", file=sys.stderr)
                error_count += 1
                continue 

            for ind,crop in enumerate(crops):
                if crop is None or crop.size == 0:
                    print(f"  Warning: Attempted to save an empty/invalid crop image for bbid {bbid}, crop_idx {ind} in CW {cwid}. Skipping.", file=sys.stderr)
                    continue

                build_name = ""
                attb = bbjson[key][bbid].get("bb_ids", [{}])[0].get("attb", {})

                if extract_dict.get("T1", False):
                    if attb.get("b+i", False): build_name+="3-"
                    elif attb.get("bold", False): build_name+="1-"
                    elif attb.get("italic", False): build_name+="2-"
                    elif attb.get("no_bi", False): build_name+="0-"
                    else: build_name+="9-" 
                
                if extract_dict.get("T2", False):
                    if attb.get("u+s", False): build_name+="3-"
                    elif attb.get("underlined", False): build_name+="1-"
                    elif attb.get("strikeout", False): build_name+="2-"
                    elif attb.get("no_us", False): build_name+="0-"
                    else: build_name+="9-" 

                if not build_name:
                    build_name = "9-" 
                
                if build_name.endswith('-'):
                    build_name = build_name[:-1]

                output_file_name = f'{build_name}_{image_file_name_only}_{bbid}_CW_{cwid}_{ind}.png'
                output_full_path = os.path.join(params["path"], output_file_name)
                
                cv2.imwrite(output_full_path, crop)
        
        prev_id=cwid
        print(f"Num crops saved for CW {cwid}: {num_crops_in_a_window_actual}. Errors in CW: {error_count}")
        
    context_window_offset+=len(cw_json[key])
    
    return context_window_offset