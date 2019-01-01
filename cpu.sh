#!/bin/bash
INTERNAL_DIR=`pwd`
WORKSPACE=$INTERNAL_DIR/workspace
PYTHON=`which python`

PS3="Please enter your choice: "
options=("clear workspace" "extract PNG from video data_src" "data_src extract faces" "data_src sort" "extract PNG from video data_dst" "data_dst extract faces" "data_dst sort by hist" "train" "convert" "converted to mp4" "quit")
select opt in "${options[@]}"
do
        case $opt in
                "clear workspace" )
                        echo -n "Clean up workspace? [Y/n] "; read workspace_ans
                        if [ "$workspace_ans" == "Y" ] || [ "$workspace_ans" == "y" ]; then
                                rm -rf $WORKSPACE
                                mkdir -p $WORKSPACE/data_src/aligned
                                mkdir -p $WORKSPACE/data_dst/aligned
                                mkdir -p $WORKSPACE/model
                                echo "Workspace has been successfully cleaned!"
                        fi
                        ;;
                "extract PNG from video data_src" )
                        echo -n "File name: "; read filename
                        echo -n "FPS: "; read fps
                        if [ -z "$fps" ]; then fps="25"; fi
                        ffmpeg -i $WORKSPACE/$filename -r $fps $WORKSPACE/data_src/%04d.png -loglevel error
                        ;;
                "data_src extract faces" )
                        echo -n "Detector? [mt | manual] "; read detector
                        $PYTHON $INTERNAL_DIR/main.py extract --input-dir $WORKSPACE/data_src --output-dir $WORKSPACE/data_src/aligned --detector $detector --debug --cpu-only
                        ;;
                "data_src sort" )
                        echo -n "Sort by? [blur | brightness | face-yaw | hue | hist | hist-blur | hist-dissim] "; read sort_method
                        $PYTHON $INTERNAL_DIR/main.py sort --input-dir $WORKSPACE/data_src/aligned --by $sort_method
                        ;;
                "extract PNG from video data_dst" )
                        echo -n "File name: "; read filename
                        echo -n "FPS: "; read fps
                        if [ -z "$fps" ]; then fps="25"; fi
                        ffmpeg -i $WORKSPACE/$filename -r $fps $WORKSPACE/data_dst/%04d.png -loglevel error
                        ;;
                "data_dst extract faces" )
                        echo -n "Detector? [mt | manual] "; read detector
                        $PYTHON $INTERNAL_DIR/main.py extract --input-dir $WORKSPACE/data_dst --output-dir $WORKSPACE/data_dst/aligned --detector $detector --debug --cpu-only
                        ;;
                "data_dst sort by hist" )
                        $PYTHON $INTERNAL_DIR/main.py sort --input-dir $WORKSPACE/data_dst/aligned --by hist
                        ;;
                "train" )
                        echo -n "Model? [ H64 (2GB+) | H128 (3GB+) | DF (5GB+) | LIAEF128 (5GB+) | LIAEF128YAW (5GB+) | MIAEF128 (5GB+) | AVATAR (4GB+) ] "; read model
                        echo -n "Show Preview? [Y/n] "; read preview
                        if [ "$preview" == "Y" ] || [ "$preview" == "y" ]; then preview="--preview"; else preview=""; fi
                        $PYTHON $INTERNAL_DIR/main.py train --training-data-src-dir $WORKSPACE/data_src/aligned --training-data-dst-dir $WORKSPACE/data_dst/aligned --model-dir $WORKSPACE/model --model $model --cpu-only $preview
                        ;;
                "convert" )
                        echo -n "Model? [ H64 (2GB+) | H128 (3GB+) | DF (5GB+) | LIAEF128 (5GB+) | LIAEF128YAW (5GB+) | MIAEF128 (5GB+) | AVATAR(4GB+) ] "; read model
                        $PYTHON $INTERNAL_DIR/main.py convert --input-dir $WORKSPACE/data_dst --output-dir $WORKSPACE/data_dst/merged --aligned-dir $WORKSPACE/data_dst/aligned --model-dir $WORKSPACE/model --model $model --ask-for-params --cpu-only
                        ;;
                "converted to mp4" )
                        echo -n "File name of destination video: "; read filename
                        echo -n "FPS: "; read fps
                        if [ -z "$fps" ]; then fps="25"; fi
                        ffmpeg -y -i $WORKSPACE/$filename -r $fps -i "$WORKSPACE/data_dst/merged/%04d.png" -map 0:a? -map 1:v -r $fps -c:v libx264 -b:v 8M -pix_fmt yuv420p -c:a aac -strict -2 -b:a 192k -ar 48000 "$WORKSPACE/result.mp4" -loglevel error
                        ;;
                "quit" )
                        break
                        ;;
                *)
                        echo "Invalid choice!"
                        ;;
        esac
done
