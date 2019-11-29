import lib


def main():
    # =====================================================================
    #                      LOAD AND ANALYZE DATA
    # =====================================================================

    train_dict = lib.load2dict(dataset='train', save=False)
    val_dict = lib.load2dict(dataset='val', save=False)

    lib.plot_patches(train_dict, max_iter=10)

    # =====================================================================
    #                           PREPROCESSING
    # =====================================================================

    # =====================================================================
    #                             DETECTION
    # =====================================================================


if __name__ == "__main__":
    main()