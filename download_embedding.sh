#!/bin/bash
echo -n "Which Files should be downloaded [All, essential, fasttext, bilingual, EQR, LID, aligned_files, MKB, MKB_pickle, BIOS]: "
read EMB

mkdir -p data/
cd data/
mkdir -p embedding/fasttext
mkdir -p embedding/Bilingual
mkdir -p embedding/LID
mkdir -p embedding/EQR
mkdir -p aligned_files
mkdir -p MKB
mkdir -p BIOS

if [ $EMB = fasttext ] || [ $EMB = All ] || [ $EMB = essential ]
then     
    cd embedding/fasttext/
    gdown --id 13UPpFVwxOf5A56sGVtoGm2yiaFd_OHTY
    gdown --id 1H1pyBdszmx3fQmS-Jmk0uXQKIlUpamYv
    gdown --id 1UlnrsbUQY-sA84wDe-_mlpFReS1gR8Nr
    gdown --id 1o-OiUq_CCzIt29n3FtzYN8JSwmy1jD5J
    cd ../../
fi

if [ $EMB = Bilingual ] || [ $EMB = All ]
then     
    cd embedding/Bilingual/
    gdown --id 1-55S7P3g49poK9-MAIO7SucwPeultVtP
    gdown --id 1-NFtneC6H_yVZp2rtG0LhuLeMm0RIKae
    gdown --id 1-NJIVL9f4Hcqy6I-MyaCWzbO_FD24qUA
    gdown --id 1-QyUyoHzGWM3X3CCxgI4DHFu-9o_QR0g
    gdown --id 1-S67MBl0KVmkJYlXncB6uz-0Cf_DBIGL
    gdown --id 1-VvePT008-8vASWehC4qV4MuYY2COvck
    gdown --id 1-Y3QoxRURw35IxkAtH--AWaxZnwF4iAY
    gdown --id 1-YPubAfjOk9JhUrTJ-hU4SrDpCygOCpD
    gdown --id 1-_ryjyxiT6RsNK8gRbZlrnXfEJgDaEL5
    gdown --id 1-drzma9uJVX5InHabOypKJaHX4HzQDba
    gdown --id 11mpq79cI1D3DXUD-cHCuzrNTO95HVzrs
    gdown --id 19vdGr3QtJ3tiguuO9zQm7n6yfl2iOSlT
    cd ../../   
fi

if [ $EMB = LID ] || [ $EMB = All ]
then 
    cd embedding/LID/
    gdown --id 1-46rPgoWIp6W_0lsNH71xXtLY0psq2rz
    gdown --id 1-8ioaku3s1golHGPsa6r4r8sWr_M-mV-
    gdown --id 1-CqJ5DGZ7q3PPaysQXbPNssRBgqM_QrQ
    gdown --id 1-Cxo_fhBM93BBT5HuyCRvekZjFXeHS1z
    gdown --id 1-E8rvz9WGEmxWreOYeGHdzXpE9DDHSA6
    gdown --id 1-Se5GCxKPohDq_h56Vc3PYAOUb9EAQyh
    gdown --id 1-W4WAyrTm-omknTZIy7nlIdrXQ5HiGkx
    gdown --id 1-XZmX_wLNfp5EJEule2ASX1ArhF4EEK4
    gdown --id 1-d6fktVL98HofJv9nEiDa0TpJjzFL8RY
    gdown --id 1-jrcYLnuNWITkDNHDmfqoXGcWl7f2v3y
    gdown --id 10AiEv56XLlPR-ayUpoSvwTyfd0bCmDAd
    gdown --id 1XHn1JjxskI6UqfQFFheFhFocglOdVfMt
    cd ../../
fi

if [ $EMB = EQR ] || [ $EMB = All ]
then 
    cd embedding/EQR/
    gdown --id 1-0_8UtLIiJv9HSxPgCCDurZ0xBFUtbt7
    gdown --id 1-9dhIkE_cDBN3ZOtZG5-090EHFpe81qj
    gdown --id 1-BwiJTysi5dKeOMRhV8PWIjeC0UK5khH
    gdown --id 1-D1F2VrQldjohaUnmdt46dF4VInZAUdu
    gdown --id 1-Eo3G97F8-ZY0H3TvWDMcVB-iZ1oBusf
    gdown --id 1-KobnSx98v9xpY3Xvg8mbpSfxyPPhGHp
    gdown --id 1-N6IYIuhC4gGC-Nz7r5prz8tcH9c4J9_
    gdown --id 1-Pnz-_vI2yurRdvPZMRfHZyib8iCCQF3
    gdown --id 1-SgVU4gH3Mv_sZEqeMHENwV25GrGzGxs
    gdown --id 1-_afs04vZ0NIwENaPAMb2-zCV5AJWdn4
    gdown --id 1-iwcTCk-qHoNr8X_clGrAdyv9zWjc9Xn
    gdown --id 1BySGQi4mT-hSAT5k5HI50588bIJa0sFS
    cd ../../
fi

if [ $EMB = aligned_files ] || [ $EMB = All ] || [ $EMB = essential ]
then 
    cd aligned_files/
    gdown --id 1-2brthWXhhAwgnFhXFqUHwJiEl2Ootq5
    gdown --id 1-3n5lMYvS_SeVgGptkwg4Y1DxpkykfY0
    gdown --id 1-9Z8v41ieg1hqOtO7Sdqigv_Gmlx5fDD
    gdown --id 15YudofLhT6-ntX080PIMOWa7j-gGRk9B
    cd ../
fi


if [ $EMB = MKB ] || [ $EMB = All ]
then 
    cd MKB
    gdown --id 144TVj4zLVoDg60PErCF56ePgOVpsV7A4
    gdown --id 14WQBE8chPj2Lsm5l3s60jOV88_RnSzbT
    gdown --id 1C-T4-WrREbBjK6WWyNf1kOKPUEHpO0JR
    gdown --id 1EIRf-B-HbxU7959gzt_xg57Jf99rr2dS
    gdown --id 1JQ8tJkPm_ZWB4RbGwWy-7r-jHnhCxyvK
    gdown --id 1NPLCwh8zgg80AuXXilHwJFBn1OUfjQdE
    gdown --id 1PSf3y7FIgDhs8HoPPRqeGA_nF8i4bz0y
    gdown --id 1Wef-t57D3AcqfbA99qWtY24CQtsfh5iO
    gdown --id 1aso9Q1AxpbP3zIsUDYQx1BdWdBMavDAz
    gdown --id 1auyH0uIy8dTDWzHAxWWWWy4Fz8rYaawO
    gdown --id 1piTy-fF8sQYE3PBq3DVjQe0X54ehSi0h
    gdown --id 1sGMWFhk4p7WZKBoTj-tK6qKTB95H7B-8    

    gdown --id 15sRpqpYS050MCunuzqtmqig6X5L0wyQI
    gdown --id 1ZOx6LANuwzXtRCjZubdcL-x2oeeopZGQ
    gdown --id 1xPePUzqjU2fG7vbz564Nq0iBGI2BdKK6
    gdown --id 1xsrXZ_jEYhaNz1BvdIEgrqyslk3hunpt
    cd ../
fi

if [ $EMB = MKB_pickle ] || [ $EMB = All ] || [ $EMB = essential ]
then 
    cd MKB
    gdown --id 1-3MJSl6VuL9nHijj2PAa0eapyjn1jMpr
    gdown --id 1-8FcaM2bAKFcj-bHRT67qCxNS3rkkZA8
    gdown --id 1-CP3up4qkhRUcpZD2ScXfH_3v323Xnm8
    gdown --id 1-ILYF6PvCzQOzOAADFS9OJ9yt1AIlyg7
    gdown --id 1-IixC4hR1mxHHu0EYJZYmoKe296qEsdu
    gdown --id 1-K1lOD_8D7jCTsk6K87TK6e0Nx7BizOR
    gdown --id 1-QWK2JNLpHGJj39gWs6v4bioI_CVkvs6
    gdown --id 1-YTJYg1FSwuKve5EvaBd8lf8lh8hikmS
    gdown --id 1-_PfyWSMtsnHJNSPCyoltljqi1MEyVVd
    gdown --id 1-b0pjOypYAKC3OshFmcyHP3PShJR5Oxc
    gdown --id 1-e2VGTVeUjyU7tjl3cBPQMUBcLVnCbTX
    gdown --id 1-iXIO3UkGcT0Jh_eRHKj1hjynaCGYUyJ
    cd ../
fi

if [ $EMB = BIOS ] || [ $EMB = All ]
then 
    cd BIOS
    gdown --id 11aL2-UJitdF3C7fe2D6g8p848-T4Em_u
    gdown --id 1Tr6ULNxKW7HqdweW5kg0q7mJyNCMG46t
    gdown --id 1i9QzLd8k0TwJ0J3hHVMI0ApW8cxuM73m
    gdown --id 1u02Ky-NJqU0FB9fDEogsWOgHB9_AodAY
    cd ../
fi