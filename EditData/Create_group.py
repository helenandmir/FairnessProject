from sklearn.neighbors import KDTree
import pandas as pd
import math
import numpy as np
import matplotlib

import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KDTree
import numpy as np
fileA ="../DataSet/Point.csv"
fileB="../DataSet/Point_radius_1000.csv"
fileC ="../DataSet/Point_group_1000_ran.csv"
fileD ="../DataSet/Point_group_1000_ran_num.csv"

df1 = pd.read_csv(fileA)
df2 = pd.read_csv(fileB)
df3 = pd.read_csv(fileC)
df4 = pd.read_csv(fileD)

center_list =[146687, 363948, 49082, 53363, 417591, 16194, 22036, 7477, 322951, 55779, 239100, 312054, 23819, 179092, 127466, 381592, 305340, 235465, 324618, 114391, 287263, 6105, 282186, 42400, 15153, 17758, 6007, 166110, 134973, 11079, 333010, 331913, 379821, 386691, 315862, 338944, 21664, 365113, 85475, 158736, 282240, 88803, 296295, 169483, 300403, 60665, 417145, 112176, 236678, 113618, 257037, 156054, 324213, 164033, 171917, 44824, 26615, 24319, 377785, 21239, 330072, 331540, 223346, 355831, 395101, 101593, 122410, 27887, 103943, 296769, 273561, 70841, 353021, 181814, 301645, 149209, 231192, 26689, 182120, 240448, 417870, 369239, 284129, 366861, 206519, 117483, 133853, 27014, 331175, 315913, 394296, 288883, 368674, 341603, 130504, 74908, 131997, 35527, 27976, 167503, 328546, 93555, 288945, 399330, 12423, 318082, 716, 23846, 387860, 375931, 359106, 313727, 284306, 59911, 72159, 129528, 312888, 351742, 11622, 16875, 338298, 73284, 58693, 384823, 173769, 162487, 344533, 222560, 315187, 130902, 268175, 253500, 51994, 48852, 89477, 102859, 371827, 168521, 89045, 62178, 203788, 66974, 371508, 99407, 263157, 154679, 278219, 139474, 405361, 250888, 14551, 318756, 146189, 318276, 117779, 369293, 370624, 226114, 224753, 419238, 238300, 107727, 191300, 389028, 243693, 413887, 126037, 162913, 174480, 61987, 402048, 316527, 109966, 196459, 264303, 419445, 179180, 121886, 251283, 66159, 393817, 230645, 284176, 158547, 415858, 155696, 171610, 44745, 21336, 102794, 383707, 343832, 398551, 84413, 139869, 173709, 323832, 1658, 293445, 140170, 357444, 125772, 248570, 287433, 388084, 247045, 262936, 20041, 143033, 52542, 275371, 224451, 256337, 348254, 124237, 7129, 371114, 105053, 96099, 269584, 17695, 41675, 377566, 121315, 329820, 410550, 371179, 273445, 81496, 237361, 109768, 66473, 272884, 174248, 241864, 399125, 75350, 53071, 336352, 13424, 210783, 118708, 38879, 358684, 101543, 148446, 21673, 211721, 122551, 147466, 365539, 345247, 145440, 120282, 286413, 247457, 92419, 405125, 8866, 288644, 41516, 379961, 175441, 31296, 253002, 207017, 67804, 276932, 386085, 140952, 288578, 264359, 308714, 414439, 49835, 205210, 288273, 286374, 323375, 110637, 105912, 25882, 419075, 226840, 245956, 63090, 81858, 342922, 112837, 66942, 411147, 311797, 15181, 28302, 372421, 168216, 358481, 181634, 377387, 423994, 169737, 156720, 186270, 182715, 407058, 315548, 118504, 366694, 305883, 151505, 221486, 264059, 390877, 287078, 412810, 263315, 280494, 397440, 86516, 207586, 4999, 275145, 23877, 418978, 143226, 185128, 333410, 27216, 338536, 414633, 86129, 336168, 160958, 142652, 155556, 202697, 303668, 45357, 78147, 76693, 186689, 82172, 257198, 38111, 47950, 322607, 92600, 80640, 331580, 151943, 147087, 195310, 196156, 155111, 28231, 420783, 80718, 302716, 152191, 94449, 359949, 111657, 180514, 381196, 357701, 250948, 159621, 160681, 305943, 283261, 278165, 285837, 212708, 329787, 162358, 370892, 226189, 31375, 328537, 305915, 162146, 30648, 370026, 29500, 258918, 206998, 220948, 344401, 191387, 211831, 45115, 151416, 81356, 66545, 279170, 148288, 109401, 336227, 202266, 6908, 413772, 409984, 227534, 313086, 399164, 343967, 349518, 337767, 182431, 275149, 311224, 63164, 397951, 411804, 65335, 1446, 291348, 290563, 89981, 25091, 339590, 130133, 57308, 152041, 88532, 384157, 160336, 84790, 154594, 307577, 354287, 407079, 124228, 132400, 360877, 273535, 181086, 4985, 327884, 115060, 44484, 273577, 322397, 279677, 288342, 170811, 139177, 194554, 246053, 348954, 390787, 356053, 216957, 262446, 134698, 115355, 205997, 189807, 367215, 100333, 31450, 305535, 269918, 126664, 266943, 406812, 391071, 30500, 257109, 103858, 165216, 186721, 221241, 247147, 139, 326807, 358765, 78113, 226699, 229696, 207713, 191167, 33760, 134787, 326969, 271350, 220975, 83347, 53232, 87339, 22327, 113003, 391987, 338938, 80174, 17027, 249843, 156533, 140121, 336483, 399079, 267803, 195851, 255721, 280002, 64252, 334657, 100522, 314912, 413063, 294516, 10189, 352024, 82338, 101143, 10395, 241395, 213179, 357059, 204038, 265603, 168685, 348234, 159813, 265998, 405739, 92048, 37849, 92956, 403901, 166465, 177604, 239160, 216420, 236792, 231889, 158018, 88920, 222845, 157947, 204761, 81860, 147411, 187050, 250019, 323698, 351565, 290716, 260192, 55289, 306407, 239499, 23575, 36058, 119890, 323182, 2166, 390883, 252937, 273804, 352973, 9755, 420350, 186854, 23986, 147147, 71380, 148791, 7050, 186486, 73617, 15724, 295116, 325956, 147654, 170912, 338908, 10402, 191020, 179198, 268433, 121783, 201835, 1483, 41149, 131914, 2558, 372449, 239837, 135324, 175968, 387133, 386361, 62018, 179441, 3001, 24082, 41677, 91560, 179005, 387251, 298695, 16308, 35617, 91350, 221008, 233898, 159580, 245256, 147890, 296978, 299719, 258821, 361331, 182439, 226520, 142642, 126111, 231037, 112247, 181600, 371865, 336123, 68780, 213782, 141228, 240632, 330160, 360939, 335085, 316009, 121668, 420565, 332331, 156806, 403316, 131291, 422613, 208131, 259675, 401602, 212671, 87108, 386538, 190830, 132521, 413876, 343411, 97123, 388784, 30588, 62550, 294786, 206991, 134384, 70731, 174656, 80625, 244166, 37127, 98652, 111489, 186249, 420085, 137130, 365354, 274961, 178120, 198212, 78105, 261120, 335776, 387096, 362377, 196209, 195918, 248408, 273846, 44687, 98129, 178723, 290121, 406770, 109269, 51113, 43267, 321129, 43871, 181307, 4553, 394736, 114939, 100524, 18772, 52319, 285183, 392217, 377286, 19563, 417790, 244525, 337412, 310733, 289350, 300416, 370145, 71366, 398424, 338410, 87705, 70887, 145391, 207620, 139640, 151156, 148550, 181104, 333535, 175205, 264181, 232013, 327424, 10665, 129162, 169595, 111145, 143977, 264478, 392486, 423887, 227396, 25742, 255676, 10237, 129441, 129839, 224039, 54115, 190726, 45138, 343386, 343544, 241960, 163284, 356102, 249136, 107344, 278626, 309595, 5400, 270884, 13474, 260027, 421615, 59802, 80767, 246950, 231782, 113363, 384400, 111269, 64336, 18214, 60761, 141013, 60027, 196025, 251473, 128300, 131730, 117393, 404156, 288956, 274245, 69038, 414881, 59486, 252688, 294093, 11479, 288154, 393238, 314437, 230634, 139551, 143004, 265219, 16168, 164748, 38882, 133208, 196073, 127654, 36665, 152372, 26593, 420919, 234375, 383739, 14349, 305518, 330145, 99233, 239452, 221019, 189208, 264504, 380127, 78413, 380423, 61877, 107688, 118101, 212735, 213257, 23526, 161848, 167461, 302353, 57933, 139194, 360399, 359869, 391857, 148347, 233589, 339157, 215278, 51618, 259302, 155211, 138530, 82112, 28957, 387640, 369803, 331775, 371221, 97995, 266424, 352276, 157125, 204352, 232455, 93438, 247492, 78084, 31635, 242649, 314162, 54584, 249956, 304184, 278546, 375678, 38893, 47491, 299024, 278961, 248390, 364334, 128766, 62459, 222615, 307658, 366883, 81485, 103263, 320281, 402943, 288918, 49890, 358210, 79033, 136841, 389746, 20869, 165281, 359830, 197725, 113040, 121341, 229878, 299330, 250235, 7744, 418428, 79451, 239439, 238947, 224014, 134271, 8965, 284859, 338151, 186061, 255890, 339579, 276945, 60744, 393412, 389159, 210135, 265664, 384067, 360219, 207783, 15148, 149171, 368456, 127774, 245007, 60569, 392094, 296842, 316030, 243407, 21740, 140396, 379086, 64722, 172052, 301017, 131550, 354829, 297034, 369153, 322773, 32451, 395372, 355832, 126742, 217382, 50209, 14556, 281306, 344258, 94839, 73431, 122974, 310483, 184487, 144407, 374623, 10362, 202097, 340328, 297849, 248091, 423330, 39897, 133908, 183494, 266852, 403548, 197498, 369224, 193676, 354009, 302783, 383909, 290566, 370886, 339446, 240731, 298927, 171586, 383789, 298542, 100128, 422225, 8576, 264, 283593, 96625, 42650, 66192, 137919, 20570, 228058, 127416, 121148, 314489, 198011, 203513, 319482, 9409, 387991, 353763, 99111, 342623, 352089, 52928, 258238, 170223, 276306]

color_list = df3.columns.tolist()
color_list.remove("ID")
df3["ID"] = center_list
df3.to_csv(fileC)
#df3.head()
df4["ID"] = center_list
df4.to_csv(fileD)
#df4.head()

dic_center_ball ={}
for id in center_list:
    dic_center_ball[id] =[]
dic_num_center_ball ={}
dic_list_num_colors ={}

# create ball
list_t = list(center_list)
tuple_color = tuple(zip(list(df1.X[list_t]), list(df1.Y[list_t]), list(df1.Z[list_t])))
tree_c = KDTree(np.array(list(tuple_color)))

for i in df1.ID:
    dist_c, ind_c = tree_c.query([[df1.X[i], df1.Y[i], df1.Z[i]]], 1)
    dic_center_ball[list_t[ind_c[0][0]]].append(i)

for c in center_list:
    dic_num_center_ball[c] = len(dic_center_ball[c])

dic_list_num_colors2 = dic_list_num_colors.copy()
for c in center_list:

    dic_list_num_colors2[c] =[c]
    for color in color_list:
        dic_list_num_colors2[c].append(len([j for j in dic_center_ball[c] if df1.Colors[j] ==color]))
for c in center_list:
    dic_list_num_colors[c] = [c]
    for color in color_list:
        if max(dic_list_num_colors2[c][1:]) == 0:
            dic_list_num_colors[c].append(0)
        else:
            dic_list_num_colors[c].append(format(len([j for j in dic_center_ball[c] if df1.Colors[j] ==color])/max(dic_list_num_colors2[c][1:]),".5f"))

for c in center_list:
    df3.loc[center_list.index(c),:] = dic_list_num_colors[c]
    df4.loc[center_list.index(c), :] = dic_list_num_colors2[c]

df3.to_csv(fileC)
df3.head()
df4.to_csv(fileD)
df4.head()
