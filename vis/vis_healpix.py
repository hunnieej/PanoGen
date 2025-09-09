import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# HEALPix 해상도 설정
nside = 16
npix = hp.nside2npix(nside)

# 예시 데이터 (픽셀 인덱스)
data = np.arange(npix)

# ERP용 θ (0~π), φ (0~2π)로 좌표 추출
theta, phi = hp.pix2ang(nside, np.arange(npix))

# ERP 평면에 해당 좌표 그리기
fig = plt.figure(figsize=(10, 5))
plt.scatter(np.degrees(phi), 90 - np.degrees(theta), c=data, cmap='viridis', s=1)
plt.title("HEALPix to ERP Mapping")
plt.xlabel("Longitude (degrees)")
plt.ylabel("Latitude (degrees)")
plt.xlim([0, 360])
plt.ylim([-90, 90])
plt.grid(True)
plt.colorbar(label="Pixel Index")
fig.savefig("healpix_to_erp.png", dpi=300)
