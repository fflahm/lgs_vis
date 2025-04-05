python query_3d.py -m ../data/room2_wxl_3/chkpnt30000.pth ^
        --load_cam --cam_json ../data/room2_wxl_3/cameras.json --cam_id 58 ^
        --ae_ckpt ../data/room2_wxl_3/best_ckpt.pth ^
        --save_path ../vis/3d/query_room2/negative.png ^
        --opac_filter --mult_by_opac --score_filter_down --score_filter_up