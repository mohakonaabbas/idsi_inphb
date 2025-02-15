import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
def mean_shift(points, bandwidth=10, max_iter=100, epsilon=1e-3):
    points = np.array(points)
    centroids = points.copy()
    
    for _ in range(max_iter):
        new_centroids = []
        for point in centroids:
            distances = np.linalg.norm(points - point, axis=1)
            within_bandwidth = points[distances < bandwidth]
            new_centroids.append(np.mean(within_bandwidth, axis=0))
        
        new_centroids = np.array(new_centroids)
        if np.linalg.norm(new_centroids - centroids) < epsilon:
            break
        centroids = new_centroids
    
    return np.unique(np.round(centroids, decimals=3), axis=0)

def detect_rotation_and_correct(image_path, ref_image_path):
    # Charger les images en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ref_image = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Détection des keypoints et des descripteurs avec SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref_image, None)
    kp2, des2 = sift.detectAndCompute(image, None)
    
    # Correspondance des descripteurs avec FLANN
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Filtrage des bonnes correspondances
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)
    
    # Affichage des correspondances
    match_img = cv2.drawMatches(ref_image, kp1, image, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("Keypoints Correspondences", match_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    show(match_img)
    
    # Calcul de la matrice de transformation si assez de points correspondent
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)
        angle = -np.arctan2(M[0, 1], M[0, 0]) * 180 / np.pi
        
        # Appliquer la rotation inverse
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        corrected_image = cv2.warpAffine(image, M_rot, (w, h))
        show(corrected_image)
        
        return corrected_image
    else:
        print("Pas assez de correspondances trouvées pour estimer la rotation.")
        return image

def detect_lines(image_path):
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    
    # Détection des bords avec Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Détection des lignes avec la transformée de Hough
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    
    # Extraction des points des lignes détectées
    if lines is not None:
        points = []
        rhos = []
        thetas = []
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            points.append([x0, y0])
            rhos.append(rho)
            thetas.append(theta)
        points = np.array(points)
        
        # Appliquer Mean Shift implémenté à partir de zéro
        maxima = mean_shift(points, bandwidth=10)
        
        # Affichage de la courbe rho-thêta
        plot_rho_theta(rhos, thetas)
        
        return maxima
    else:
        return []

def plot_rho_theta(rhos, thetas):
    plt.figure()
    plt.scatter(thetas, rhos, color='b', marker='o')
    plt.xlabel("Theta (radians)")
    plt.ylabel("Rho (pixels)")
    plt.title("Courbe rho-thêta de la transformée de Hough")
    plt.grid()
    plt.show()


#
def return_time(image_path : str,
                image_reference : str = ""):
    

    computed_time = None
    
    return computed_time

def preprocess_image(image : np.array, display = False):
    ret2,thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    # Define a kernel for dilation ( for small needle)
    # kernel = np.ones((3, 3), np.uint8)
    # thresh = cv2.dilate(thresh, kernel, iterations=1)


    edges = cv2.Canny(thresh,100,127)
    if display:
        show(edges)
    return edges

def show(image, name =" image"):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def get_longest_contour(edges):
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:  # If no contours are found, return None
        return None
    
    # Initialize variables to track the most elongated contour
    longest_contour = None
    max_perimeter = 0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)  # Calculate the perimeter (closed contour)
        if perimeter > max_perimeter:
            max_perimeter = perimeter
            longest_contour = contour

    # Create a blank image to draw the most elongated contour
    contour_img = np.zeros_like(edges)
    if longest_contour is not None:
        cv2.drawContours(contour_img, [longest_contour], -1, 255, thickness=1)

    return longest_contour, max_perimeter, contour_img
def get_associated_countours(edges,
                             rho,
                             theta,
                             threshold,
                             display = True):
    
    y,x = np.where(edges >0 )
    points = np.vstack((x,y)).T
    distances = np.abs( np.cos(theta) * (x-rho*np.cos(theta)) + np.sin(theta)*(y - rho*np.sin(theta)))
    active_points_mask =distances  < threshold
    contours = points[active_points_mask]


    
        # Create a blank image to draw contours
    contour_img = np.zeros_like(edges)
    contour_img[contours[:,1],contours[:,0]]=255
        # Define a kernel for dilation ( for small needle)
    kernel = np.ones((3, 3), np.uint8)
    contour_img = cv2.dilate(contour_img, kernel, iterations=2)
    contour_img = cv2.erode(contour_img, kernel, iterations=2)

    longest_contour, max_perimeter, contour_img = get_longest_contour(contour_img)

        # Show image using OpenCV
    if display:
        show(contour_img)

    return longest_contour, max_perimeter, contour_img


def return_detected_lines(image : np.array, display = True, thres = 1):
    lines = cv2.HoughLines(image,1,np.pi/360,80)
    associated_contours = []
    perimeters = []
    if display:
        display_image = np.zeros_like(image) #image.copy()
        display_image = cv2.cvtColor(display_image,cv2.COLOR_GRAY2BGR)
        
    
    if display:
        plt.scatter(lines[:,0,0],lines[:,0,1])
        plt.show()

    # Try to cluster the lines
    norms = np.linalg.norm(lines.copy().squeeze(), axis = 1).astype(np.uint16)
    # norms = np.sort(norms)
    initial_order = np.argsort(norms)
    ordered_norms = norms[initial_order]-np.roll(norms[initial_order],1)>10
    (indexes,) = np.where(ordered_norms)
    line_1 = lines[initial_order[0:max(indexes[1]-1,1)]].mean(axis=0)
    line_2 = lines[initial_order[indexes[1]:indexes[2]-1]].mean(axis=0)
    line_3 = lines[initial_order[indexes[2]:]].mean(axis=0)
    filtered_lines = np.vstack((line_1,line_2,line_3))[:,np.newaxis,:]
    # Lazy center calculus ( can be more robust )

    rho_1,rho_2,a1,a2, b1, b2 = -line_2[0,0],-line_3[0,0],np.cos(line_2[0,1]),np.cos(line_3[0,1]), np.sin(line_2[0,1]),np.sin(line_3[0,1])
    y_center = (a1*rho_2-a2*rho_1)/(b1*a2-a1*b2)
    x_center = -(b1*y_center+rho_1)/(a1+1e-12) 

    center = np.array([x_center,y_center])
    for line in filtered_lines:
        for rho, theta in line:
            longest_contour, max_perimeter, contour_img = get_associated_countours(image, rho, theta=theta, threshold= thres)
            associated_contours.append(longest_contour)
            perimeters.append(max_perimeter)
            if display:
                a = np.cos(theta)
                b = np.sin(theta)   
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(display_image,(x1,y1),(x2,y2),(0,0,255),2)
        
    

    if display:
        cv2.circle(display_image,center=center.astype(np.int32).tolist(),radius=3,color=(255),thickness=-1)
        show(display_image)


    return filtered_lines , perimeters, center




def compute_time(filtered_lines , perimeters, center):

    # Identify second , minute and hour needle,
    order = np.argsort(perimeters)
    second_needle = filtered_lines[order[2]]
    minutes_needle = filtered_lines[order[1]]
    hours_needle = filtered_lines[order[0]]
    # Remettre tout dans le repere de l'horloge
    transform = np.eye(3)
    transform[0,0]=0
    transform[0,1]=1
    transform[1,0]=-1
    transform[1,1]=0
    transform[0,2] = -center[0]
    transform[1,2] = -center[1]

    
    s = np.array([np.cos(second_needle[0,1]), np.sin(second_needle[0,1]),0])
    m = np.array([np.cos(minutes_needle[0,1]), np.sin(minutes_needle[0,1]),0])
    h = np.array([np.cos(hours_needle[0,1]), np.sin(hours_needle[0,1]),0])
    s_p = np.dot(transform,s)
    m_p = np.dot(transform,m)
    h_p = np.dot(transform,h)

    # Calculer les angles 
    s_angle = np.atan2(s_p[0],-s_p[1])
    m_angle = np.atan2(m_p[0],-m_p[1])
    h_angle = np.atan2(h_p[0],-h_p[1])
    f = lambda x : 2*np.pi -(x-np.pi/2) if x>np.pi/2 else x
    secondes = int(f(s_angle) * 60 /(2*np.pi))
    minutes = int(f(m_angle) * 60 /(2*np.pi))
    heures = int(f(h_angle) * 12 /(2*np.pi))



    # Faire les conversions
    print( f"Il est {heures} h - {minutes} minutes - {secondes} secondes" )
    return (heures, minutes, secondes)



if __name__ == "__main__":

    for _ in range(10):
        tic = time.time()
        image_path = "data_images/database/clockD.png"  # Remplace par ton chemin d'image
        ref_image_path = "data_images/database/clockE.png"  # Image de référence
        corrected_image = detect_rotation_and_correct(image_path, ref_image_path)
        # img = cv2.imread(image_path,0)
        edges = preprocess_image(corrected_image)
        filtered_lines , perimeters, center = return_detected_lines(edges,thres=3)

        compute_time(filtered_lines , perimeters, center)
        toc = time.time()
        print(toc-tic)
    
    # corrected_image = detect_rotation_and_correct(image_path, ref_image_path)
    # maxima = detect_lines(image_path)
    # print("Maxima locaux après Mean Shift:", maxima)
