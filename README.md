# TreeDetection
random orchard images from Google maps are passed down along with their scale bar so that an automatic tree detection is enabled from aerial imagery. The framework can detect the trees and create a log for the location of the trees’ centerpoint as well as the crown size for images in any size. The training starts with the classification of each pixel with a tree/non-tree label. Then, the old method of template matching is equipped to heuristically sweep over all the images and output the templates that match the tree crown with the highest score and minimal overlap.

This work is established based on the work by Yang et al:
Yang, Lin, et al. "Tree detection from aerial imagery." Proceedings of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems. 2009.
