import sys
import os

import pyocr
import pyocr.builders
import pyautogui
import cv2

from PIL import Image

from time import sleep

# 参考
# https://qiita.com/onaka_yurusugi/items/7fe2bacb7ede88eadd1b

# PATH of Tesseract https://github.com/UB-Mannheim/tesseract/wiki
TESSERACT_PATH = "C:\\Users\\user\\AppData\\Local\\Programs\\Tesseract-OCR" 
TESSDATA_PATH = TESSERACT_PATH + '\\tessdata'
TESSERACT_LAYOUT = 10 # tesseract-ocr/tesseract/doc/tesseract.1.asc の --psm N

os.environ["PATH"] += os.pathsep + TESSERACT_PATH
os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH


tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
# The tools are returned in the recommended order of usage
tool = tools[0]
print("tool '%s'" % (tool.get_name()))
# Ex: Will use tool 'libtesseract'

langs = tool.get_available_languages()
print("Available languages: %s" % ", ".join(langs))
lang = langs[0]
print("Will use lang '%s'" % (lang))
# Ex: Will use lang 'fra'
# Note that languages are NOT sorted in any way. Please refer
# to the system locale settings for the default language
# to use.

# 範囲指定のためのマウスカーソル座標取得関数。メッセージボックスの左上隅と右下隅で囲まれた範囲をスクリーンショット
def PosGet():
    # 3秒待ってからカーソル位置の座標を取得
    print("左上隅の座標を取得します")
    sleep(3)
    x1, y1 = pyautogui.position()
    print(str(x1) + "," + str(y1))

    # 3秒待ってからカーソル位置の座標を取得
    print("右下隅の座標を取得します")
    sleep(3)
    x2, y2 = pyautogui.position()

    # PyAutoGuiのregionの仕様のため、相対座標を求める
    x2 -= x1
    y2 -= y1

    print(str(x2) + "," + str(y2))

    return(x1, y1, x2, y2)

# スクリーンショット撮影 → グレースケール → 画像を拡大
def ScreenShot(x1, y1, x2, y2, misc):
    sc = pyautogui.screenshot(region=(x1, y1, x2, y2))
    sc.save("images/"+misc+str(x1)+"_"+str(y1)+'.jpg')
    # あとは画像拡大してみましょうか グレースケールも有効？ OpenCVにも頼ってみよう
    img = cv2.imread("images/"+misc+str(x1)+"_"+str(y1)+'.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmp = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("images/"+misc+str(x1)+"_"+str(y1)+'.jpg', tmp)

def prefix(txt, misc):
    if misc == "height":
        dic4height = {"6":"0", "0":"0", "l":"1", "I":"1", "1":"1", "の":"2", "2":"2", "3":"3", "4":"4", "ら":"5", "5":"5",
                       "ム":"6", "6":"6", "7":"7", "8":"8", "9":"9"}
        try:
            txt = dic4height[txt]
        except:
            return 1, 0
    elif misc == "fuel":
        dic4fuel = {"o":"0", "O":"0", "S":"5", "Z":"2"}
        flightovers = set(["We","W0","WO","Wo","Ws","Vea","ee","W","Ve","WA","WA,","RA","WA.","RA,","RA.","Ua","Ua,","Ua.","","","","","","",""])
        if txt[-3:] == "m/s":
            txt = txt[:-3]
        elif txt[-4:]=="m /s" or txt[-4:]=="mi/s":
            txt = txt[:-4]
        elif txt in flightovers:
            txt = "114514"
        for w in dic4fuel:
            txt = txt.replace(w, dic4fuel[w])
    print(txt)

    if txt.isdecimal():
        return 0, int(txt)
    else:
        return 1, 0

def capture(misc, pos_str):
    txt = tool.image_to_string(
        Image.open("images/"+misc+pos_str),
        lang="eng",
        builder=pyocr.builders.TextBuilder(tesseract_layout=TESSERACT_LAYOUT)
    )
    return prefix(txt, misc)

def controll(isFuelEmpty, isFlightOver, isPrevFuelEmpty):
    if isFuelEmpty or isPrevFuelEmpty or isFlightOver:
        pyautogui.press("space")
        print("pressed Space")
        sleep(1)


def main():
    print("Input 'dev' if you need x-y information.")
    if input() == "dev":
        x1, y1, x2, y2 = PosGet()

    positions = [[733,190,20,25,"height"],[755,190,20,25,"height"],[777,190,20,25,"height"],
                [202,688,45,12,"fuel"]]

    isPrevFuelEmpty = True
    isFlightOver = False
    print("starting in 5 seconds...")
    sleep(5)
    pyautogui.press("t")
    print("pressed T")
    while True:
        dat = [0]*len(positions)
        for i in range(len(positions)):
            x1, y1, x2, y2 = positions[i][0], positions[i][1], positions[i][2], positions[i][3]
            misc = positions[i][4]

            flag = 1
            while flag == 1:
                ScreenShot(x1, y1, x2, y2, misc)
                flag, num = capture(misc, str(x1)+"_"+str(y1)+'.jpg')
                dat[i] = num
            
            height = dat[0]*100 + dat[1]*10 + dat[2]
            isFuelEmpty = dat[3]==0
            isFlightOver = dat[3]==114514
        print("height:",height)
        print("isFuelEmpty:",isFuelEmpty)
        print("isFlightOver:",isFlightOver)
        print("-"*20)

        controll(isFuelEmpty, isFlightOver, isPrevFuelEmpty)
        isPrevFuelEmpty = isFuelEmpty

        if isFlightOver:
            break

if __name__ == "__main__":
    main()