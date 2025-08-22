# # echo backup test_no_grad
# rsync -Pauv hkqai:/home/chenzihao/workspace/cc2cc_test5/data/test_no_grad/ ~/workspace/2025.1/data/test_no_grad
# rsync -Pauv dgx1:/home/chenzihao/workspace/cc2cc_test5/data/test_no_grad/ ~/workspace/2025.1/data/test_no_grad
# rsync -Pauv ~/workspace/2025.1/data/test_no_grad/ hkqai:/home/chenzihao/workspace/cc2cc_test5/data/test_no_grad
# rsync -Pauv ~/workspace/2025.1/data/test_no_grad/ dgx1:/home/chenzihao/workspace/cc2cc_test5/data/test_no_grad

# # echo backup test
# rsync -Pauv hkqai:/home/chenzihao/workspace/cc2cc_test5/data/test/ ~/workspace/2025.1/data/test
# rsync -Pauv dgx1:/home/chenzihao/workspace/cc2cc_test5/data/test/ ~/workspace/2025.1/data/test
# rsync -Pauv ~/workspace/2025.1/data/test/ hkqai:/home/chenzihao/workspace/cc2cc_test5/data/test
# rsync -Pauv ~/workspace/2025.1/data/test/ dgx1:/home/chenzihao/workspace/cc2cc_test5/data/test

# echo backup checkpoints
# rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' hkqai:/home/chenzihao/workspace/cc2cc_test5/checkpoints/ ~/workspace/2025.1/checkpoints
# rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' dgx1:/home/chenzihao/workspace/cc2cc_test5/checkpoints/ ~/workspace/2025.1/checkpoints
# rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' ~/workspace/2025.1/checkpoints/ hkqai:/home/chenzihao/workspace/cc2cc_test5/checkpoints
# rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' ~/workspace/2025.1/checkpoints/ dgx1:/home/chenzihao/workspace/cc2cc_test5/checkpoints

rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' dgx1:/home/chenzihao/workspace/cc2cc_test5/checkpoints/ /home/chenzihao/workspace/cc2cc_test5/checkpoints
rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' /home/chenzihao/workspace/cc2cc_test5/checkpoints/ dgx1:/home/chenzihao/workspace/cc2cc_test5/checkpoints

# # echo backup grids_dft

# # check if the directory exists
# if [ ! -d "/media/dhem/Elements/data/grids_dft" ]; then
#     echo "Directory /media/dhem/Elements/data/grids_dft does not exist. Perhaps you need to mount the Elements drive?"
#     exit 1
# else
#     echo "Directory /media/dhem/Elements/data/grids_dft exists. Proceeding with rsync."
#     rsync -Pauv dgx1:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ /media/dhem/Elements/data/grids_dft
#     rsync -Pauv hkqai:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ /media/dhem/Elements/data/grids_dft
#     rsync -Pauv /media/dhem/Elements/data/grids_dft/ dgx1:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft
#     rsync -Pauv /media/dhem/Elements/data/grids_dft/ hkqai:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft
# fi
