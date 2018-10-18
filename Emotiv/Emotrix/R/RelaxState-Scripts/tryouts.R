#script para limpiar data (antes de wavelets)
#Mario Barrientos - 13039
install.packages("readr")
library(readr)
quality = 7
setwd("~/Documents/Emotrix/emotrix/2017/Data")


# SUJETO 1 - ALVARO GALINDO
alvaro_galindo <- read_csv("alvaro_galindo.csv",col_types = cols(`Exact Time` = col_double()))
# View(alvaro_galindo)
#2. agarrar data en donde todos los sensores tengan calidad mayor o igual a 10 (good signal according to emotiv)
clean1 <-alvaro_galindo[(alvaro_galindo$F3_Quality >=quality & alvaro_galindo$F4_Quality >= quality & alvaro_galindo$AF3_Quality >= quality & alvaro_galindo$AF4_Quality >= quality & alvaro_galindo$O1_Quality >= quality & alvaro_galindo$O2_Quality >= quality),]
# View(clean1)
#remover variable del environment para tener memoria suficiente
remove(alvaro_galindo)


#SUJETO 2 - ANDRES BARRIENTOS
andres_barrientos <- read_csv("andres_barrientos.csv",col_types = cols(`Exact Time` = col_double()))
# View(andres_barrientos)
#2. agarrar data en donde todos los sensores tengan calidad mayor o igual a 10 (good signal according to emotiv)
clean2 <-andres_barrientos[(andres_barrientos$F3_Quality >=quality & andres_barrientos$F4_Quality >= quality & andres_barrientos$AF3_Quality >= quality & andres_barrientos$AF4_Quality >= quality & andres_barrientos$O1_Quality >= quality & andres_barrientos$O2_Quality >= quality),]
# View(clean2)
remove(andres_barrientos)

#SUJETO 3 - CHRISTAL SHARINA
christal_sharina <- read_csv("christal_sharina.csv",col_types = cols(`Exact Time` = col_double()))
# View(christal_sharina)
clean3 <-christal_sharina[(christal_sharina$F3_Quality >=quality & christal_sharina$F4_Quality >= quality & christal_sharina$AF3_Quality >= quality & christal_sharina$AF4_Quality >= quality & christal_sharina$O1_Quality >= quality & christal_sharina$O2_Quality >= quality),]
# View(clean3)
# nrow(clean3)
remove(christal_sharina)

#SUJETO 4 - DANIEL OROZCO
daniel_orozco <- read_csv("daniel_orozco.csv",col_types = cols(`Exact Time` = col_double()))
# View(daniel_orozco)
clean4 <-daniel_orozco[(daniel_orozco$F3_Quality >=quality & daniel_orozco$F4_Quality >= quality & daniel_orozco$AF3_Quality >= quality & daniel_orozco$AF4_Quality >= quality & daniel_orozco$O1_Quality >= quality & daniel_orozco$O2_Quality >= quality),]
# View(clean4)
# nrow(clean4)
remove(daniel_orozco)

#SUJETO 5 - DIDIER
didier <- read_csv("didier.csv",col_types = cols(`Exact Time` = col_double()))
# View(didier)
clean5 <-didier[(didier$F3_Quality >=quality & didier$F4_Quality >= quality & didier$AF3_Quality >= quality & didier$AF4_Quality >= quality & didier$O1_Quality >= quality & didier$O2_Quality >= quality),]
# View(clean5)
# nrow(clean5)
remove(didier)

#SUJETO 6 - JAVIER FONG
javier_fong <- read_csv("javier_fong.csv",col_types = cols(`Exact Time` = col_double()))
# View(javier_fong)
clean6 <-javier_fong[(javier_fong$F3_Quality >=quality & javier_fong$F4_Quality >= quality & javier_fong$AF3_Quality >= quality & javier_fong$AF4_Quality >= quality & javier_fong$O1_Quality >= quality & javier_fong$O2_Quality >= quality),]
# View(clean6)
# nrow(clean6)
remove(javier_fong)

#SUJETO 7 - JUAN CANTEO
juan_canteo <- read_csv("juan_canteo.csv",col_types = cols(`Exact Time` = col_double()))
# View(juan_canteo)
clean7 <-juan_canteo[(juan_canteo$F3_Quality >=quality & juan_canteo$F4_Quality >= quality & juan_canteo$AF3_Quality >= quality & juan_canteo$AF4_Quality >= quality & juan_canteo$O1_Quality >= quality & juan_canteo$O2_Quality >= quality),]
# View(clean7)
# nrow(clean7)
remove(juan_canteo)


#SUJETO 8 - MONICA SANDOVAL
monica_sandoval <- read_csv("monica_sandoval.csv",col_types = cols(`Exact Time` = col_double()))
# View(monica_sandoval)
clean8 <-monica_sandoval[(monica_sandoval$F3_Quality >=quality & monica_sandoval$F4_Quality >= quality & monica_sandoval$AF3_Quality >= quality & monica_sandoval$AF4_Quality >= quality & monica_sandoval$O1_Quality >= quality & monica_sandoval$O2_Quality >= quality),]
# View(clean8)
# nrow(clean8)
remove(monica_sandoval)


#SUJETO 9 - ROBERTO CHIROY 
roberto_chiroy <- read_csv("roberto_chiroy.csv",col_types = cols(`Exact Time` = col_double()))
# View(roberto_chiroy)
clean9 <-roberto_chiroy[(roberto_chiroy$F3_Quality >=quality & roberto_chiroy$F4_Quality >= quality & roberto_chiroy$AF3_Quality >= quality & roberto_chiroy$AF4_Quality >= quality & roberto_chiroy$O1_Quality >= quality & roberto_chiroy$O2_Quality >= quality),]
# View(clean9)
# nrow(clean9)
remove(roberto_chiroy)


#SUJETO 10 - SUSSETH FERNANDEZ
susseth_fernandez <- read_csv("susseth_fernandez.csv",col_types = cols(`Exact Time` = col_double()))
# View(susseth_fernandez)
clean10 <-susseth_fernandez[(susseth_fernandez$F3_Quality >=quality & susseth_fernandez$F4_Quality >= quality & susseth_fernandez$AF3_Quality >= quality & susseth_fernandez$AF4_Quality >= quality & susseth_fernandez$O1_Quality >= quality & susseth_fernandez$O2_Quality >= quality),]
# View(clean10)
# nrow(clean10)
remove(susseth_fernandez)


#SUJETO 11 - WILLIAM FUENTES
william_fuentes <- read_csv("william_fuentes.csv",col_types = cols(`Exact Time` = col_double()))
# View(william_fuentes)
clean11 <-william_fuentes[(william_fuentes$F3_Quality >=quality & william_fuentes$F4_Quality >= quality & william_fuentes$AF3_Quality >= quality & william_fuentes$AF4_Quality >= quality & william_fuentes$O1_Quality >= quality & william_fuentes$O2_Quality >= quality),]
# View(clean11)
# nrow(clean11)
remove(william_fuentes)

#Juntar toda la data
total <- rbind(clean1, clean2, clean3, clean4, clean5, clean6, clean7, clean8, clean9, clean10, clean11)
# View(total)
#Filtrar data que solo tiene info del estado de relajacion
rsdata <- total[(total$Emotion =="RELAX" | total$Emotion=="NON-RELAX"),]
# nrow(rsdata)
# View(rsdata)
rsdata$`Selected Emotion`<-NULL
nonrelax <- nrow(rsdata[(rsdata$Emotion == "NON-RELAX"),])
relax <- nrow(rsdata[(rsdata$Emotion == "RELAX"),])
info <- data.frame(relax,nonrelax,nrow(rsdata))
View(info)

