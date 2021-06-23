%得到了room1_1 ~ room40_3； Room1 ~ Room30给无监督，Room31~38给训练，39~40给测试集；
%21种噪声 1-15给无监督， 16-19种给ASR训练集，20/21给测试集；
room_config = '/home/guoyifan/MultiChannel/config/room_sensor_config_all.txt';
train_100_scp = "/home/guoyifan/MultiChannel/scp/train_clean_100.scp";
out_path = "/home/guoyifan/MultiChannel/data/train_clean_100/";
out_100_scp = out_path + "asr.scp";
out_100_overlap_scp = out_path + "overlap_direction/ol_asr.scp";
out_100_no_ol_scp = out_path + "no_overlap/no_ol_asr.scp";
noise_path = '/home/guoyifan/MultiChannel/data/my_point_noises/';

log_file = '/home/guoyifan/MultiChannel/log/asr.log';
% flog = fopen(log_file,'w');

% train_100_scp = "/home/guoyifan/MultiChannel/src/train_clean_100.scp";
augment_factor = 3;

ncores = 25;
Mypar = parpool(ncores);

[param,value]=textread(room_config,'%9s%[^\n]','commentstyle','matlab');
q = strrep(' '' ',' ','');
for p=1:length(param),
    eval([param{p} '=[' value{p} ']' q ';']);
end
clear param value p;

ROOM = zeros([40,3,17]);
for itx = 1:40
    for ity = 1:3
        ROOM(itx,ity,:) = eval(["room"+int2str(itx)+"_"+int2str(ity)]);
    end
end
%得到了room1_1 ~ room40_3； Room1 ~ Room30给无监督，Room31~38给训练，39~40给测试集；
%21种噪声 1-15给无监督， 16-19种给ASR训练集，20/21给测试集；
%1-37:noise 16-19 first half, no overlap;
%38-48:noise 16-19 first half, overlap
%49-57:noise 16-21 second half, no
%58-60:noise 16-21 second half, overlap

%1-16:noise 16-19 first half, no overlap;
%17-20:noise 16-19 first half, overlap
%21-24:noise 16-21 second half, no
%24-25:noise 16-21 second half, overlap
%Unsurpervised
scplist = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"];
parfor it = 1:ncores
    c_1 = floor(it/26)+1;
    c_2 = mod(it,26);
    if c_2 == 0
        c_1 = c_1-1;
        c_2 = 26;
    end
    fdin = fopen(train_100_scp + scplist(c_1) + scplist(c_2),'r');
    flog = fopen(log_file + scplist(c_1) + scplist(c_2), 'w');
    if ((it<=16) || (20<it && it<=24))
        fdout = fopen(out_100_scp+ scplist(c_1) + scplist(c_2),'w');
    else
        fdout_ol = fopen(out_100_overlap_scp+ scplist(c_1) + scplist(c_2),'w');
        fdout_no_ol = fopen(out_100_no_ol_scp+ scplist(c_1) + scplist(c_2),'w');
    end
    
    cnt_360 = 0;
    while ~feof(fdin)
        line = fgetl(fdin);
        line = split(line, " ");
        wavname = line{6}; % /workspace/data/asr/LibriSpeech//LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac
        %
        temp_name = strsplit(wavname,"/");
        temp_name = temp_name{end};
        temp_name = strsplit(temp_name,".");
        temp_name = temp_name{1};
        for i = 1:augment_factor
            out_name = temp_name; % 103-1240-0000<wav-name>
            room_id = randi(30);
            sensor_id = randi(3);
            room_name = "room"+int2str(room_id)+"_"+int2str(sensor_id);
            room = ROOM(room_id,sensor_id,:);
            room_name = room_name.split("_");
            room_name = room_name.join("-");
            
            out_name = out_name + "_" + room_name;  % 103-1240-0000<wav-name>_room3-2;
            if it <= 20
                noise_num = randi(4)+15; % 16~19
            else
                noise_num = randi(6)+15; % 16~21
            end
            
            if noise_num<10
                noise_name = strcat('00',int2str(noise_num));
            else
                noise_name = strcat('0',int2str(noise_num));
            end
            
            [wav,    fs] = audioread(wavname);
            noise_n = strcat(noise_path,noise_name,'.wav');
            [wav_n,fs_n] = audioread(noise_n); % 010.wav
            wav_size = size(wav);
            wav_n_size = size(wav_n);
            if it<=20
                n_offset = randi(floor(wav_n_size(1)/2)-wav_size(1));
            else
                n_offset = randi(floor(wav_n_size(1)/2)-wav_size(1))+ceil(wav_n_size(1)/2);
            end
            
            wav_n = wav_n(n_offset:n_offset+wav_size(1)-1,:);
            
            if mod(cnt_360,50)==0
                disp(char(int2str(cnt_360)+" of part "+int2str(it)+" have been done!\n"));
                fprintf(flog, "%s\n", int2str(cnt_360)+" of part "+int2str(it)+" have been done!\n");
            end
            cnt_360 = cnt_360 + 1;
            if ((it<=16) || (20<it && it<=24))
                [time,HH, pos, dir]=my_roomsimove(room,wav_size(1), fs, 4);
                [time_n,HH_n, pos_n, dir_n]=my_roomsimove(room,wav_size(1),fs, 9);
                if wav_size(2)>1
                    wav = (wav(:,1) + wav(:,2))/2;
                end
                if wav_n_size(2)>1
                    wav_n = (wav_n(:,1) + wav_n(:,2))/2;
                end
                x=roomsimove_apply(time,HH,wav',fs);
                x_n=roomsimove_apply(time_n,HH_n,wav_n',fs);
                x_power = sum(sum(x.^2));
                x_n_power = sum(sum(x_n.^2));
                ratio = (randi(10))^2;
                x = x * sqrt(ratio * x_n_power / x_power);
                snr = 10 * log(ratio) / log(10);
                x_out = x + x_n;
                x_out = x_out/max(max(max(x_out)),-min(min(x_out)));
                % 103-1240-0000<wav-name>_room3-2; _source-2.1-3.2-2.5
                out_name = out_name + "_source-"+num2str(round(pos(1),1))+"-"+num2str(round(pos(2),1))+"-"+num2str(round(pos(3),1));
                % 103-1240-0000<wav-name>_room3-2_source-2.1-3.2-2.5;_noise015_2.2-2.5-1.9_snr.wav;
                out_name = out_name + "_noise"+noise_name+"_"+num2str(round(pos_n(1),1))+"-"+num2str(round(pos_n(2),1))+"-"+num2str(round(pos_n(3),1))+"_"+num2str(round(snr))+".wav";
                fprintf(fdout,"%s %s\n",temp_name, out_path+out_name);
                
                audiowrite(char(out_path+out_name), x_out', fs);
            else
                [time,HH, pos, dir]=my_roomsimove(room,wav_size(1), fs, 4);
                [time_n_ol,HH_n_ol, pos_n_ol, dir_n_ol]=my_roomsimove(room,wav_size(1),fs, 9, dir);
                [time_n,HH_n, pos_n, dir_n]=my_roomsimove(room,wav_size(1),fs, 9);
                if wav_size(2)>1
                    wav = (wav(:,1) + wav(:,2))/2;
                end
                if wav_n_size(2)>1
                    wav_n = (wav_n(:,1) + wav_n(:,2))/2;
                end
                x=roomsimove_apply(time,HH,wav',fs);
                x_n_ol=roomsimove_apply(time_n_ol,HH_n_ol,wav_n',fs);
                x_n=roomsimove_apply(time_n,HH_n,wav_n',fs);
                x_power = sum(sum(x.^2));
                x_n_ol_power = sum(sum(x_n_ol.^2));
                x_n_power = sum(sum(x_n.^2));
                ratio = (randi(10))^2;
                x_no = x * sqrt(ratio * x_n_power / x_power);
                x_ol = x * sqrt(ratio * x_n_ol_power / x_power);
                snr = 10 * log(ratio) / log(10);
                x_out = x_no + x_n;
                x_out_ol = x_ol + x_n_ol;
                x_out = x_out/max(max(max(x_out)),-min(min(x_out)));
                x_out_ol = x_out_ol/max(max(max(x_out_ol)),-min(min(x_out_ol)));
                % 103-1240-0000<wav-name>_room3-2; _source-2.1-3.2-2.5
                out_name = out_name + "_source-"+num2str(round(pos(1),1))+"-"+num2str(round(pos(2),1))+"-"+num2str(round(pos(3),1));
                % 103-1240-0000<wav-name>_room3-2_source-2.1-3.2-2.5;_noise015_2.2-2.5-1.9_snr.wav;
                out_name_no_ol = out_name + "_noise"+noise_name+"_"+num2str(round(pos_n(1),1))+"-"+num2str(round(pos_n(2),1))+"-"+num2str(round(pos_n(3),1))+"_"+num2str(round(snr))+".wav";
                out_name_ol = out_name + "_noise"+noise_name+"_"+num2str(round(pos_n_ol(1),1))+"-"+num2str(round(pos_n_ol(2),1))+"-"+num2str(round(pos_n_ol(3),1))+"_"+num2str(round(snr))+".wav";
                fprintf(fdout_ol,"%s %s\n",temp_name, out_path+"/overlap_direction/"+out_name_ol);
                fprintf(fdout_no_ol,"%s %s\n",temp_name, out_path+"/no_overlap/"+out_name_no_ol);
                audiowrite(char(out_path+"/overlap_direction/"+out_name_ol), x_out_ol', fs);
                audiowrite(char(out_path+"/no_overlap/"+out_name_no_ol), x_out', fs);
            end
            
        end
    end
    fclose(fdin);
    fclose(flog);
    if ((it<=16) || (20<it && it<=24))
        fclose(fdout);
    else
        fclose(fdout_ol);
        fclose(fdout_no_ol);
    end
end
delete(Mypar);
