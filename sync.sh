if [[ -n $SSH_CONNECTION ]]; then
    rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' dgx1:/home/chenzihao/workspace/cc2cc_test5/checkpoints/ /home/chenzihao/workspace/cc2cc_test5/checkpoints
    rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' /home/chenzihao/workspace/cc2cc_test5/checkpoints/ dgx1:/home/chenzihao/workspace/cc2cc_test5/checkpoints
else
    echo backup test
    rsync -Pauv hkqai:/home/chenzihao/workspace/cc2cc_test5/data/test/ ~/workspace/2025.1/data/test
    rsync -Pauv dgx1:/home/chenzihao/workspace/cc2cc_test5/data/test/ ~/workspace/2025.1/data/test
    rsync -Pauv ~/workspace/2025.1/data/test/ hkqai:/home/chenzihao/workspace/cc2cc_test5/data/test
    rsync -Pauv ~/workspace/2025.1/data/test/ dgx1:/home/chenzihao/workspace/cc2cc_test5/data/test

    echo backup checkpoints
    rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' hkqai:/home/chenzihao/workspace/cc2cc_test5/checkpoints/ ~/workspace/2025.1/checkpoints
    rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' dgx1:/home/chenzihao/workspace/cc2cc_test5/checkpoints/ ~/workspace/2025.1/checkpoints
    rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' ~/workspace/2025.1/checkpoints/ hkqai:/home/chenzihao/workspace/cc2cc_test5/checkpoints
    rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' ~/workspace/2025.1/checkpoints/ dgx1:/home/chenzihao/workspace/cc2cc_test5/checkpoints

    # echo backup grids_dft

    echo "Directory ~/workspace/2025.1/data/grids_dft exists. Proceeding with rsync."
    rsync -Pauv dgx1:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ ~/workspace/2025.1/data/grids_dft
    rsync -Pauv hkqai:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ ~/workspace/2025.1/data/grids_dft
    rsync -Pauv ~/workspace/2025.1/data/grids_dft/ dgx1:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft
    rsync -Pauv ~/workspace/2025.1/data/grids_dft/ hkqai:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft
fi
