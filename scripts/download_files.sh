# ./serialized_dict.pkl
# https://drive.google.com/file/d/1RvgZMELiIdm_peFKdJiHbIwVtgu55X6z/view?usp=sharing
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RvgZMELiIdm_peFKdJiHbIwVtgu55X6z' -O serialized_dict.pkl

# ./models/yolo/names.pkl
# https://drive.google.com/file/d/1C4etbyFnMxfxaWvJ8HNqO_cwyrKxccX_/view?usp=sharing
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1C4etbyFnMxfxaWvJ8HNqO_cwyrKxccX_' -O ./models/yolo/names.pkl

# ./models/yolo/colors.pkl
# https://drive.google.com/file/d/1ImrRT-ERNYw3_fW-qVfl4JLQv55NvSY3/view?usp=sharing
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ImrRT-ERNYw3_fW-qVfl4JLQv55NvSY3' -O ./models/yolo/colors.pkl

# ./models/yolo/yolov3-tiny.pt
# https://drive.google.com/file/d/1k39G0CrSGvW23AuyJGIuVI1WKQSg1X3s/view?usp=sharing
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1k39G0CrSGvW23AuyJGIuVI1WKQSg1X3s' -O ./models/yolo/yolov3-tiny.pt

# DOESN't work because yolov3.pt is > 100m
# ./models/yolo/yolov3.pt
# https://drive.google.com/file/d/1TcMNp_OdBoILbaPSIx0fF-LoywWV0Cc6/view?usp=sharing 
# wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TcMNp_OdBoILbaPSIx0fF-LoywWV0Cc6' -O ./models/yolo/yolov3.pt

# Download yolov3.pt in 3 parts 
# https://drive.google.com/file/d/1AhwBfLkx4bPfYNqpR8bnbrHimMK58jCu/view?usp=sharing
# https://drive.google.com/file/d/1aCUdegTxIZzUY5pD7zEUOeyBat5d-7S9/view?usp=sharing
# https://drive.google.com/file/d/1wEroQeWD_B_iyeE6UjSNdrUvGNjp2Jtt/view?usp=sharing
# https://drive.google.com/file/d/1jQQH93i3QB7iq1v23boeJkL3_KzJFR5A/view?usp=sharing
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AhwBfLkx4bPfYNqpR8bnbrHimMK58jCu' -O ./models/yolo/yolov3.tar.part.1
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aCUdegTxIZzUY5pD7zEUOeyBat5d-7S9' -O ./models/yolo/yolov3.tar.part.2
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wEroQeWD_B_iyeE6UjSNdrUvGNjp2Jtt' -O ./models/yolo/yolov3.tar.part.3
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jQQH93i3QB7iq1v23boeJkL3_KzJFR5A' -O ./models/yolo/yolov3.tar.part.4
# Combine into 1 single tar 
cat ./models/yolo/yolov3.tar.part* > ./models/yolo/yolov3.tar
# uncompress this tar 
tar xf ./models/yolo/yolov3.tar -C ./models/yolo/