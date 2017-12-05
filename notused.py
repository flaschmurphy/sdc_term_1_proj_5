
    ###############################################################
    #
    # Test adding boundary boxes
    #
    ###############################################################
    #image = imread('./test_images/bbox-example-image.jpg')
    #bboxes = [((837, 507), (1120, 669))]
    #result = draw_boxes(image, bboxes)
    #imshow(result)()
    #plt.show()

    ###############################################################
    #
    # Muck around with boxing using cv2 image templates
    #
    ###############################################################
    #bboxs = template_matching()
    #image = imread('./test_images/bbox-example-image.jpg')
    #image = draw_boxes(image, bboxs)
    #imshow(image)()
    #plt.show()

    ###############################################################
    #
    # Show color histogram for color space
    #
    ###############################################################
    #img = imread('test_images/cutout1.jpg')
    #rhist, ghist, bhist, hist_features, bin_centers = color_hist(img, cspace='YCrCb', vis=True)
    #fig = plt.figure(figsize=(12,3))
    #plt.subplot(131)
    #plt.bar(bin_centers, rhist[0])
    #plt.xlim(0, 256)
    #plt.title('R Histogram')
    #plt.subplot(132)
    #plt.bar(bin_centers, ghist[0])
    #plt.xlim(0, 256)
    #plt.title('G Histogram')
    #plt.subplot(133)
    #plt.bar(bin_centers, bhist[0])
    #plt.xlim(0, 256)
    #plt.title('B Histogram')
    #plt.show()


    ###############################################################
    #
    # Explore RGB and HSV color spaces in 3D
    #
    ###############################################################
    ##files = glob.glob('./test_images/[0-9][0-9].png')
    #files = glob.glob('./test_images/*.png')
    #for f in files:
    #    print(f)
    #    img = imread(f)
    #    
    #    # Select a small fraction of pixels to plot by subsampling it
    #    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    #    img_small = cv2.resize(img, (np.int(img.shape[1] / scale),
    #        np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
    #    
    #    # Convert subsampled image to desired color space(s)
    #    img_small_RGB = img_small
    #    img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
    #    img_small_HLS = cv2.cvtColor(img_small, cv2.COLOR_RGB2HLS)
    #    img_small_YCR = cv2.cvtColor(img_small, cv2.COLOR_RGB2YCrCb)
    #    img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting
    #    
    #    # Plot and show
    #    plot3d(img_small_RGB, img_small_rgb, axis_labels=list("RGB"))
    #    plt.show()
    #    
    #    plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
    #    plt.show()

    #    plot3d(img_small_HLS, img_small_rgb, axis_labels=list("HLS"))
    #    plt.show()

    #    plot3d(img_small_YCR, img_small_rgb, axis_labels=list("YCrCb"))
    #    plt.show()


    ###############################################################
    #
    # Resizing images
    #
    ###############################################################

    #image = imread('test_images/cutout1.jpg')
    #feature_vec = bin_spatial(image, color_space='LUV', size=(32, 32))
    ## Plot features
    #plt.plot(feature_vec)
    #plt.title('Spatially Binned Features')
    #plt.show()
    ## Question - can we scale down even further? Consider this when training classifier

    ###############################################################
    #
    # Test the HOG function
    #
    ###############################################################

    # Generate a random index to look at a car image
    #d = load_data(car_or_not='car', random=True)
    #fname, image, idx = d[0]

    ## Convert to gray before sending to HOG
    #gray = rgb2gray(image)

    ## Define HOG parameters
    #orient = 9
    #pix_per_cell = 8
    #cell_per_block = 2

    ## Call our function with vis=True to see an image output
    #features, hog_image = get_hog_features(
    #        gray, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)

    ## Plot the examples
    #fig = plt.figure()
    #plt.subplot(121)
    #imshow(image, cmap='gray')()
    #plt.title('Example Car Image')
    #plt.subplot(122)
    #plt.imshow(hog_image, cmap='gray')
    #plt.title('HOG Visualization')
    #plt.show()


    ###############################################################
    #
    # Sanity check the data
    #
    ###############################################################
    #_ = load_data(sanity=True)

