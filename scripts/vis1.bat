set CAM=00000
python vis_3d_feature.py -m ../data/room1_wxl_3/chkpnt30000.pth ^
            --load_cam --cam_json ../data/room1_wxl_3/cameras.json --cam_id %CAM% ^
            --opac_filter