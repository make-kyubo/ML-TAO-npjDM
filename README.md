# ML-TAO-npjDM

Five python codes (to build ML models predicting each inflammatory sign) and sample data (images and labels)  
Each code generates .xlsx files which contain prediction results.  
Additional codes will be needed to calculate CAS (Clinical Activity Score) and to measure the performance of each model or whole system.

## sample_data/
### Redness_of_eyelids/
Cropped images each containing eyelid  
The photos with left eye are flipped horizontally to look like right eye.

### Redness_of_conjunctiva/
#### medial/
Cropped images each containing medial canthus  
The photos with left eye are flipped horizontally to look like right eye.

#### lateral/
Cropped images each containing lateral canthus  
The photos with left eye are flipped horizontally to look like right eye.

### Swelling_of_eyelids/
Cropped images each containing eyelid  
The photos with left eye are flipped horizontally to look like right eye.

### Inflammation_of_caruncle/
Cropped images each containing medial canthus  
The photos with left eye are flipped horizontally to look like right eye.

### Conjunctival_edema/
Cropped images each containing lateral canthus  
The photos with left eye are flipped horizontally to look like right eye.

### scoring_sample.xlsx
CAS (Clinical Activity Score) scored by three ophthalmologists  
- Each ophthalmologist's diagnosis (for each sign, for each eye of all patients)
- Three ophthalmologists' agreed final diagnosis (for each sign, for each eye of all patients)
- Existence of two types of orbital pain (spontaneous retrobulbar pain and pain on gaze)

## Redness_of_eyelids.py
Submodels (per-doc) and aggregating model to predict the redness of the eyelids  

## Redness_of_conjunctiva.py
Submodels (per-doc) to predict the redness of the conjunctiva
To build the submodels, the code loads two kinds of images (medial & lateral).  
Additional codes will be needed to choose the majority of results.

## Swelling_of_eyelids.py
Submodels (per-doc) to predict the swelling of the eyelids  
Additional codes will be needed to choose the majority of results.

## Inflammation_of_caruncle.py
Submodels (per-doc) and aggregating model to predict the inflammation of the caruncle  

## Conjunctival_edema.py
Submodels (per-doc) to predict the conjunctival edema  
Additional codes will be needed to choose the majority of results.
