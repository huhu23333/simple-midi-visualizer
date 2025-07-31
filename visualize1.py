import os
import mido
import cv2
import numpy as np
import time
import random

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
midi_path = os.path.join(current_dir, "Wings of Courage.mid")

mid = mido.MidiFile(midi_path)

max_note = 108
note_shift = 12
start_blank_len = 24*128
end_blank_len = 24*128

screen = (1280, 720)

line_width_piano = 1
line_width_note = 2
line_color_piano = (0,0,0)
piano_black_note_color = (20,20,20)
piano_white_note_color = (240, 240, 240)
black_key_height = 48
white_key_height = 80

note_y_scale = 2.5
piano_x_scale = 1.05

particle_speed = 15
particle_spread = 0.5
particle_decay = 0.90
particle_size = 2
particle_amount = 7
particle_masks_number = 3

use_particle = True
use_glow = True

output_path = os.path.join(current_dir, "output.mp4")
fps = 30

""" color_list = [(254, 26, 106), (183, 26, 255), (255, 190, 25), (55, 226, 146), 
              (0, 231, 255), (36, 201, 152), (245, 234, 32), (100, 208, 92),
              (140, 179, 234), (247, 253, 51), (216, 123, 220)
              ] """
color_list = [(234, 28, 116), (116, 97, 185), (231, 209, 5), (3, 224, 221), 
              (139, 81, 142), (120, 174, 52), (186, 38, 186), (205, 160, 22),
              (243, 160, 0), (0, 232, 161), (72, 185, 100)]
""" color_list = [(241, 70, 57), (205, 241, 0), (50, 201, 20), (107, 241, 231), 
              (127, 67, 255), (241, 127, 200), (170, 212, 170), (222, 202, 170), 
              (241, 201, 20), (80, 80, 80), (202, 50, 127)] """

color_list = [(color[2], color[1], color[0]) for color in color_list]

blank_state_frame = [0 for _ in range(max_note)]
tracks_time_note_state = []
#time_len_types = []
for track in mid.tracks:
    tracks_time_note_state.append([blank_state_frame]*start_blank_len)
    on_note_list = []
    for msg in track:
        if type(msg) == mido.messages.messages.Message:
            md = msg.dict()
            """ if md['time'] not in time_len_types:
                time_len_types.append(md['time']) """
            if md['time'] != 0:
                state_frame = [0 for _ in range(max_note)]
                for on_note in on_note_list:
                    state_frame[on_note] = 2
                for _ in range(md['time']):
                    tracks_time_note_state[-1].append(state_frame.copy())
            if md["type"]=='note_on':
                on_note_list.append(md['note']-note_shift)
            if md["type"]=='note_off':
                tracks_time_note_state[-1][-1][md['note']-note_shift] = 3
                if md['note']-note_shift in on_note_list:
                    del on_note_list[on_note_list.index(md['note']-note_shift)]
                else:
                    print(f"Warning: note {md['note']-note_shift} should be on")
    if md['time'] != 0:
        state_frame = [0 for _ in range(max_note)]
        for on_note in on_note_list:
            state_frame[on_note] = 2
        tracks_time_note_state[-1].append(state_frame.copy())
    for note in range(max_note):
        for time_step in reversed(range(len(tracks_time_note_state[-1]))):
            if tracks_time_note_state[-1][time_step][note] != 0:
                if time_step==0 or tracks_time_note_state[-1][time_step-1][note] != 2:
                    tracks_time_note_state[-1][time_step][note] = 1
    
#print(sorted(time_len_types))

for track_index in reversed(range(len(tracks_time_note_state))):
    if len(tracks_time_note_state[track_index]) == start_blank_len:
        del tracks_time_note_state[track_index]

max_len = max([len(track) for track in tracks_time_note_state])

for track_index in range(len(tracks_time_note_state)):
    tracks_time_note_state[track_index] += [state_frame]*(max_len-len(tracks_time_note_state[track_index])+end_blank_len)

# tracks_time_note_state : [0 vocal, 1 other1（开头伴奏）, 2 other2（music box）, 3 other3（主歌伴奏）,4 piano1（主伴奏）, 5 piano2（主歌旋律）, 6 guitar1（木吉他）, 7 guitar2（电吉他）, 8 bass, 9 drums, 10 other4（副歌伴奏）]

""" order_transform_list = [0,3,4,7,8,10,6,5,2,1,9]
tracks_time_note_state = [tracks_time_note_state[index] for index in order_transform_list] """
tracks_time_note_state = np.array(tracks_time_note_state) # (144, 250, 255)


def count_consecutive_true(arr, start):
    sub_arr = arr[start:]
    if sub_arr.size == 0:
        return 0
    first_false_index = np.argmax(~sub_arr)
    if ~sub_arr[first_false_index]:
        return first_false_index
    else:
        return sub_arr.size

def add_glow_effect(image, mask, glow_color=(255, 255, 255), gaussian_size=63, alpha=1.5, sigma=0):
    color_layer = np.zeros_like(image, dtype=np.float32)
    color_layer[:,:] = glow_color
    weight_mask_3d = cv2.merge([mask]*3)
    glow_layer = weight_mask_3d * color_layer
    blurred_glow = cv2.GaussianBlur(glow_layer, (gaussian_size, gaussian_size), sigmaX=sigma, sigmaY=sigma).astype(np.float32)/255
    image_float = image.astype(np.float32) + blurred_glow * alpha
    np.clip(image_float, 0, 255, out=image_float)
    image[:] = image_float.astype(np.uint8)

note_to_x_ratio_total = screen[0]/max_note
note_to_x_ratio_white = note_to_x_ratio_total*12/7
note_type_in_a_octave_list = [0,1,0,1,0,0,1,0,1,0,1,0]
note_to_white = [0,0,1,1,2,3,3,4,4,5,5,6]
def draw_piano(image, glow_mask, particle_masks, state_frame, on_color):
    for note in range(max_note):
        if note_type_in_a_octave_list[note%12] == 0:
            x_left = int(note_to_white[note%12] * note_to_x_ratio_white + (note//12)*12*note_to_x_ratio_total)
            x_right = int((note_to_white[note%12] + 1) * note_to_x_ratio_white + (note//12)*12*note_to_x_ratio_total)
            key_color = on_color if state_frame[note] != 0 else piano_white_note_color
            pt1 = (x_left, screen[1]-white_key_height-1)
            pt2 = (x_right, screen[1]-1)
            cv2.rectangle(image, pt1=pt1, pt2=pt2, color=key_color, thickness=-1)
            cv2.rectangle(image, pt1=pt1, pt2=pt2, color=line_color_piano, thickness=line_width_piano)
    for note in range(max_note):
        if note_type_in_a_octave_list[note%12] == 1:
            x_left = int(note * note_to_x_ratio_total)
            x_right = int((note + 1) * note_to_x_ratio_total)
            key_color = on_color if state_frame[note] != 0 else piano_black_note_color
            pt1 = (x_left, screen[1]-white_key_height-1)
            pt2 = (x_right, screen[1]-white_key_height+black_key_height-1)
            cv2.rectangle(image, pt1=pt1, pt2=pt2, color=key_color, thickness=-1)
            cv2.rectangle(image, pt1=pt1, pt2=pt2, color=line_color_piano, thickness=line_width_piano)
    for note in range(max_note):
        if state_frame[note] != 0:
            mask_temp = np.zeros(image.shape[:2], dtype=np.uint8)
            if note_type_in_a_octave_list[note%12] == 0:
                x_left = int(note_to_white[note%12] * note_to_x_ratio_white + (note//12)*12*note_to_x_ratio_total)
                x_right = int((note_to_white[note%12] + 1) * note_to_x_ratio_white + (note//12)*12*note_to_x_ratio_total)
                pt1 = (x_left, screen[1]-white_key_height-1)
                pt2 = (x_right, screen[1]-1)
                cv2.rectangle(mask_temp, pt1=pt1, pt2=pt2, color=255, thickness=-1)
                note -= 1
                x_left = int(note * note_to_x_ratio_total)
                x_right = int((note + 1) * note_to_x_ratio_total)
                pt1 = (x_left, screen[1]-white_key_height-1)
                pt2 = (x_right, screen[1]-white_key_height+black_key_height-1)
                cv2.rectangle(mask_temp, pt1=pt1, pt2=pt2, color=0, thickness=-1)
                note += 2
                x_left = int(note * note_to_x_ratio_total)
                x_right = int((note + 1) * note_to_x_ratio_total)
                pt1 = (x_left, screen[1]-white_key_height-1)
                pt2 = (x_right, screen[1]-white_key_height+black_key_height-1)
                cv2.rectangle(mask_temp, pt1=pt1, pt2=pt2, color=0, thickness=-1)
                note -= 1
            else:
                x_left = int(note * note_to_x_ratio_total)
                x_right = int((note + 1) * note_to_x_ratio_total)
                pt1 = (x_left, screen[1]-white_key_height-1)
                pt2 = (x_right, screen[1]-white_key_height+black_key_height-1)
                cv2.rectangle(mask_temp, pt1=pt1, pt2=pt2, color=255, thickness=-1)
            np.maximum(glow_mask, mask_temp, out=glow_mask)
            if use_particle:
                x_left = int(note * note_to_x_ratio_total)
                x_right = int((note + 1) * note_to_x_ratio_total)
                random_range_x = [x_left, x_right]
                for particle_mask_index in range(len(particle_masks)):
                    particle_mask_params = particle_masks[particle_mask_index][1]
                    random_range_y = [screen[1]-white_key_height-particle_mask_params["particle_speed"], screen[1]-white_key_height]
                    for _ in range(particle_mask_params["particle_amount"]):
                        pt = (random.randint(*random_range_x), random.randint(*random_range_y))
                        cv2.circle(particle_masks[particle_mask_index][0], pt, particle_mask_params["particle_size"], 1.0, -1)
    pt1 = (0, screen[1]-white_key_height-3)
    pt2 = (screen[0], screen[1]-white_key_height-1)
    cv2.rectangle(image, pt1=pt1, pt2=pt2, color=on_color, thickness=-1)
    pt1 = (0, screen[1]-white_key_height-1)
    pt2 = (screen[0], screen[1]-white_key_height+3)
    cv2.rectangle(glow_mask, pt1=pt1, pt2=pt2, color=255, thickness=-1)
    if use_particle:
        for particle_mask_index in range(len(particle_masks)):
            color_layer = np.zeros_like(image, dtype=np.float32)
            color_layer[:,:] = on_color
            weight_mask_3d = cv2.merge([particle_masks[particle_mask_index][0]]*3)
            image_float = image.astype(np.float32) + weight_mask_3d * color_layer
            np.clip(image_float, 0, 255, out=image_float)
            image[:] = image_float.astype(np.uint8)
            np.maximum(glow_mask, (particle_masks[particle_mask_index][0]*255).astype(np.uint8), out=glow_mask)

            particle_mask_params = particle_masks[particle_mask_index][1]
            translation_matrix = np.float32([[1, 0, particle_mask_params["particle_spread"]], [0, 1, -particle_mask_params["particle_speed"]]])
            particle_masks[particle_mask_index][0][:] = cv2.warpAffine(
                src=particle_masks[particle_mask_index][0], 
                M=translation_matrix,
                dsize=(image.shape[1], image.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            ) * particle_mask_params["particle_decay"]

def draw_notes(image, glow_mask, track_time_note_state, time_step, note_color):
    note_line_color = (int(note_color[0]*0.7),int(note_color[1]*0.7),int(note_color[1]*0.7))
    check_len = int((screen[1]-white_key_height)*note_y_scale)
    track_len = len(track_time_note_state)
    check_region = track_time_note_state[time_step : min(time_step+check_len, track_len-1)]
    check_len = len(check_region)
    if check_len == 0:
        return
    check_start_indexes = np.where(check_region == 1)
    started_start_indexes = np.where(np.expand_dims(check_region[0], axis=0) == 2)
    #print(check_start_indexes)
    for check_start_index in zip(np.concatenate((check_start_indexes[0], started_start_indexes[0]), axis=0), np.concatenate((check_start_indexes[1], started_start_indexes[1]), axis=0)):
        note = check_start_index[1]
        note_len = count_consecutive_true(check_region[:, note] == 2, check_start_index[0]+1)+1
        note_start_y = screen[1] - white_key_height - int(check_start_index[0]/note_y_scale)
        note_end_y = screen[1] - white_key_height - int((check_start_index[0] + note_len)/note_y_scale)
        x_left = int(note * note_to_x_ratio_total)
        x_right = int((note + 1) * note_to_x_ratio_total)
        pt1 = (x_left, note_end_y)
        pt2 = (x_right, note_start_y)
        cv2.rectangle(image, pt1=pt1, pt2=pt2, color=note_color, thickness=-1)
        cv2.rectangle(image, pt1=pt1, pt2=pt2, color=note_line_color, thickness=line_width_note)
        cv2.rectangle(glow_mask, pt1=pt1, pt2=pt2, color=255, thickness=-1)

track_number = len(tracks_time_note_state)
track_len = len(tracks_time_note_state[0])

def translation_matrix_params_function(time_step_float, step_len):
    frame = time_step_float/step_len
    pt1 = (0, -64) #10, -64
    pt2 = (2, -7)  #2, -7
    t1 = 60
    t2 = 90
    t3 = 135
    if frame<t1:
        tx, ty = pt1
    elif frame<t2:
        tx = (pt1[0]-pt2[0])/((t1-t2)**2)*((frame-t2)**2)+pt2[0]
        ty = pt1[1]
    elif frame<t3:
        tx = pt2[0]
        ty = (pt1[1]-pt2[1])/((t2-t3)**2)*((frame-t3)**2)+pt2[1]
    else:
        tx, ty = pt2
    return tx, ty

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式常用编解码器
out = cv2.VideoWriter(output_path, fourcc, fps, screen)

particle_masks_tracks = [[[np.zeros((screen[1], screen[0]), np.float32),{
    "particle_speed": particle_speed,
    "particle_spread": particle_spread * i,
    "particle_decay": particle_decay,
    "particle_size": particle_size,
    "particle_amount": particle_amount
}] for i in np.linspace(-1, 1, particle_masks_number)] for j in range(track_number)]
time_step_float = 0
step_len = 24*32*40/60/30
info_output_step_len = 100
info_output_step_count = 0
color_show_t1 = 15
color_show_t2 = 45
color_alpha = 0.7
while time_step_float < track_len:
    t = time.time()
    time_step = int(time_step_float)
    image = np.zeros((screen[1], screen[0], 3), np.uint8)
    tx, ty = translation_matrix_params_function(time_step_float, step_len)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    time_step_float += step_len
    for track_index in reversed(range(track_number)): # track_number
        cv2.warpAffine(
            src=image, 
            M=translation_matrix,
            dsize=(image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
            dst=image
        )
        track_time_note_state = tracks_time_note_state[track_index]
        color = color_list[track_index]
        particle_masks = particle_masks_tracks[track_index]
        glow_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        draw_notes(image, glow_mask, track_time_note_state, time_step, color)
        draw_piano(image, glow_mask, particle_masks, track_time_note_state[time_step], color)
        if use_glow:
            add_glow_effect(image, glow_mask, color)
        if time_step_float/step_len < color_show_t1:
            alpha = color_alpha
            color_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.rectangle(color_mask, (0, screen[1]-white_key_height), (screen[0], screen[1]), 255, -1)
            color_layer = np.zeros_like(image)
            color_layer[:] = color
            image[color_mask == 255] = cv2.addWeighted(
                image, 1 - alpha, 
                color_layer, alpha, 
                0
            )[color_mask == 255]
        elif time_step_float/step_len < color_show_t2:
            alpha = (color_show_t2 - time_step_float/step_len) / (color_show_t2 - color_show_t1) * color_alpha
            color_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.rectangle(color_mask, (0, screen[1]-white_key_height), (screen[0], screen[1]), 255, -1)
            color_layer = np.zeros_like(image)
            color_layer[:] = color
            image[color_mask == 255] = cv2.addWeighted(
                image, 1 - alpha, 
                color_layer, alpha, 
                0
            )[color_mask == 255]

    cv2.imshow("", image)
    out.write(image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    if info_output_step_count%info_output_step_len == 0:
        spf = time.time()-t
        print(f"Rendering {time_step/track_len*100:.2f}% [{time_step}/{track_len}] [{spf:.3f} spf] (about {(track_len-time_step)/step_len*spf:.1f} s left)")
    info_output_step_count += 1
    
out.release()

""" octave_count = max_note // 12 + (1 if max_note%12 != 0 else 0)
note_to_x_ratio_total = screen[0]/max_note
note_type_in_a_octave_list = [0,1,0,1,0,0,1,0,1,0,1,0]

def draw_piano_pixel(x, state_frame, on_color):
    note = x / note_to_x_ratio_total
    int_note = int(note)
    if x%note_to_x_ratio_total <= line_width_piano/2 or note_to_x_ratio_total-x%note_to_x_ratio_total <= line_width_piano/2:
        return line_color_piano
    if state_frame[int_note] == 2 or state_frame[int_note] == 1:
        return on_color
    if note_type_in_a_octave_list[int_note%12] == 0:
        return piano_white_note_color
    else:
        return piano_black_note_color

for state_frame in tracks_time_note_state[0]:
    image = np.zeros((screen[1], screen[0], 3), np.uint8)
    for x in range(screen[0]):
        color = draw_piano_pixel(x, state_frame, (240, 80, 80))
        image[:,x] = color
    cv2.imshow("", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break """

#print([len(track) for track in tracks_time_note_state])

""" print(len(tracks_time_note_state))
print([len(track) for track in tracks_time_note_state]) """
""" for state_frame in tracks_time_note_state[1]:
    temp = ""
    for note in state_frame:
        temp += "1" if note else "0"
    print(temp) """

            
