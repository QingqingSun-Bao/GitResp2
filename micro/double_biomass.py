import pandas as pd
from sqlalchemy import create_engine
from Load_Save import LoadDict

'''double biomass'''


def spe_bio(data, ex, spec, zuhe):
    bio_1 = 0
    bio_2 = 0
    data = data[data["顺序"] == str(float(ex))]
    for zh in zuhe:
        dt1 = data[data["样地号"] == zh]
        dt2 = dt1[dt1["物种"] == spec[0]]["干重g"]
        dt3 = dt1[dt1["物种"] == spec[1]]["干重g"]
        # print(float(dt2))
        bio_1 = bio_1 + float(dt2)
        bio_2 = bio_2 + float(dt3)
    return bio_1, bio_2


def main():
    engine = create_engine('mysql+pymysql://root:Sun!@#123@106.75.247.108:3306/nchenjiang?charset=utf8')
    path = "C:/Users/97899/Desktop/N/"
    two_ex = LoadDict(path + "Zuhe/two_ex.txt")
    zuhe_plot = LoadDict(path + "Zuhe/Zuhe_plot.txt")
    two_nested_ex=LoadDict(path + "Zuhe/two_nested_ex.txt")
    spec = ["黄囊苔草", "糙隐子草"]
    # "黄囊苔草"
    hu_nested = {}; hu_no_nested={}
    bio_hu = {}
    yang_nested = {};yang_no_nested={}
    bio_yang = {}

    for year in range(2008, 2020):
        hu_nested[year] = {}; bio_hu[year] = {}
        yang_nested[year] = {}; bio_yang[year] = {}
        hu_no_nested[year]={};yang_no_nested[year]={}
        df_bio = pd.read_sql(str(year), con=engine)
        print(year)
        for ex in range(1, 39):
            # nest
            if [year, ex] in two_nested_ex[tuple(["黄囊苔草", "糙隐子草"])]:
                hu_nested[year][ex],yang_nested[year][ex] = spe_bio(df_bio, ex, spec, zuhe_plot[year][ex])
            else:
                hu_nested[year][ex] = 0
                yang_nested[year][ex]=0
            # all
            if [year, ex] in two_ex[tuple(["黄囊苔草", "糙隐子草"])] :
                bio_hu[year][ex],bio_yang[year][ex] = spe_bio(df_bio, ex, spec, zuhe_plot[year][ex])
            else:
                bio_hu[year][ex] = 0
                bio_yang[year][ex]=0
            # no_nest
            if ([year, ex] not in two_nested_ex[("黄囊苔草", "糙隐子草")])and([year, ex] in two_ex[tuple(["黄囊苔草", "糙隐子草"])]):
                hu_no_nested[year][ex],yang_no_nested[year][ex] = spe_bio(df_bio, ex, spec, zuhe_plot[year][ex])
            else:
                hu_no_nested[year][ex] = 0
                yang_no_nested[year][ex]=0
    write=pd.ExcelWriter(path+"caAndhu/ca_hu_biomass.xls")
    pd.DataFrame(hu_nested).to_excel(write,sheet_name="hu_nested")
    pd.DataFrame(bio_hu).to_excel(write, sheet_name="hu_bio")
    pd.DataFrame(yang_nested).to_excel(write, sheet_name="ca_nested")
    pd.DataFrame(bio_yang).to_excel(write, sheet_name="ca_bio")
    pd.DataFrame(hu_no_nested).to_excel(write, sheet_name="hu_no_nested")
    pd.DataFrame(yang_no_nested).to_excel(write, sheet_name="ca_no_nested")
    write.save()
    write.close()


main()
