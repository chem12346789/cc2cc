if [[ -n $SSH_CONNECTION ]]; then
    case "$(hostname)" in
    "manage.hkcorp.internal")
        echo "Syncing files from manage.hkqai"
        for directory in validate validate_hkqai validate_hkqai_done; do
            echo "Syncing ${directory} from dgx1 to manage.hkqai"
            rsync -Pauv "dgx1:/home/chenzihao/workspace/cc2cc_test5/${directory}/" \
                "/home/chenzihao/workspace/cc2cc_test5/${directory}/"
            echo "Syncing ${directory} from manage.hkqai to dgx1"
            rsync -Pauv "/home/chenzihao/workspace/cc2cc_test5/${directory}/" \
                "dgx1:/home/chenzihao/workspace/cc2cc_test5/${directory}/"
        done
        rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' dgx1:/home/chenzihao/workspace/cc2cc_test5/checkpoints/ /home/chenzihao/workspace/cc2cc_test5/checkpoints
        rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' /home/chenzihao/workspace/cc2cc_test5/checkpoints/ dgx1:/home/chenzihao/workspace/cc2cc_test5/checkpoints
        echo "copy grids_dft from dgx1 to manage.hkqai"
        rsync -Pauv dgx1:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ /home/chenzihao/workspace/cc2cc_test5/data/grids_dft/
        echo "copy grids_dft from data_generation to manage.hkqai.hpc"
        rsync -Pauv data_generation:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ /home/chenzihao/workspace/cc2cc_test5/data/grids_dft/
        echo "copy grids_dft from dgx2 to manage.hkqai.hpc"
        rsync -Pauv dgx2:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ /home/chenzihao/workspace/cc2cc_test5/data/grids_dft/
        echo "copy grids_dft from manage.hkqai.hpc to dgx1"
        rsync -Pauv /home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ dgx1:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft/
        echo "copy grids_dft from manage.hkqai.hpc to data_generation"
        rsync -Pauv /home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ data_generation:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft/
        echo "copy grids_dft from manage.hkqai.hpc to dgx2"
        rsync -Pauv /home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ dgx2:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft/
        ;;
    "dgxa100svr01")
        echo "Syncing files from dgxa100svr01"
        for directory in validate validate_hkqai; do
            echo "Syncing ${directory} from hkqai to dgx1"
            rsync -Pauv "hkqai:/home/chenzihao/workspace/cc2cc_test5/${directory}/" \
                "/home/chenzihao/workspace/cc2cc_test5/${directory}/"
            echo "Syncing ${directory} from dgx1 to hkqai"
            rsync -Pauv "/home/chenzihao/workspace/cc2cc_test5/${directory}/" \
                "hkqai:/home/chenzihao/workspace/cc2cc_test5/${directory}/"
        done
        echo "copy grids_dft from hkqai to dgxa100svr01"
        rsync -Pauv hkqai:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ /home/chenzihao/workspace/cc2cc_test5/data/grids_dft/
        echo "copy grids_dft from data_generation to dgxa100svr01"
        rsync -Pauv data_generation:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ /home/chenzihao/workspace/cc2cc_test5/data/grids_dft/
        echo "copy grids_dft from dgxa100svr01 to hkqai"
        rsync -Pauv /home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ hkqai:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft
        echo "copy grids_dft from dgxa100svr01 to data_generation"
        rsync -Pauv /home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ data_generation:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft
        rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' hkqai:/home/chenzihao/workspace/cc2cc_test5/checkpoints/ /home/chenzihao/workspace/cc2cc_test5/checkpoints
        rsync -Pauv --exclude 'trash' --exclude 'backup' --exclude 'saved' /home/chenzihao/workspace/cc2cc_test5/checkpoints/ hkqai:/home/chenzihao/workspace/cc2cc_test5/checkpoints
        ;;
    "user-MZ73-LM1-000")
        echo "Syncing files from user-MZ73-LM1-000"
        echo "copy grids_dft from hkqai to user-MZ73-LM1-000"
        rsync -Pauv hkqai:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ /home/chenzihao/workspace/cc2cc_test5/data/grids_dft/
        echo "copy grids_dft from data_generation to user-MZ73-LM1-000"
        rsync -Pauv dgx1:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ /home/chenzihao/workspace/cc2cc_test5/data/grids_dft/
        echo "copy grids_dft from user-MZ73-LM1-000 to hkqai"
        rsync -Pauv /home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ dgx1:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft
        echo "copy grids_dft from user-MZ73-LM1-000 to data_generation"
        rsync -Pauv /home/chenzihao/workspace/cc2cc_test5/data/grids_dft/ hkqai:/home/chenzihao/workspace/cc2cc_test5/data/grids_dft
        ;;
    esac
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
