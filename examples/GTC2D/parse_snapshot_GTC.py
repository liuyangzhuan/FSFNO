# STFNO Copyright (c) 2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S.Dept. of Energy) and the University of
# California, Berkeley.  All rights reserved.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.

import torch
import torch.nn as nn

def parse_snapshot(snapshot_path):
    raw = np.genfromtxt(snapshot_path)
    base_filename = os.path.basename(snapshot_path)
    step = int(re.findall(r"snap0+(\d+).out", base_filename)[0])
    check_sum = np.sum(raw)
    ret = {}
    if np.isnan(check_sum):
        ret["Status"] = 2
        ret["info"] = "{} in file {}".format(error_codes[3], snapshot_path)
    nspecies = int(raw[0])
    nfield = int(raw[1])
    nvgrid = int(raw[2])
    mpsi_plus = int(raw[3])
    mt_plus = int(raw[4])
    mtoroidal = int(raw[5])
    poloidata = raw[7+nspecies*6*mpsi_plus+nspecies*4*nvgrid:7 +
                    nspecies*6*mpsi_plus+nspecies*4*nvgrid+mt_plus*mpsi_plus*(nfield+2)]
    start = 7+nspecies*6*mpsi_plus+nspecies*4*nvgrid+mt_plus*mpsi_plus*(nfield+2)
    fluxdata = raw[start:start+nfield*mtoroidal*mt_plus]
    poloidata_3d = poloidata.reshape((nfield+2, mpsi_plus, mt_plus))
    phi_2d = poloidata_3d[0, :, :]
    apara_2d = poloidata_3d[1, :, :]
    flowi = poloidata_3d[22, :, :]
    densityi = poloidata_3d[10, :, :]
    pressureipara = poloidata_3d[21, :, :]
    pressureiperp = poloidata_3d[7, :, :]
    sfluidne = poloidata_3d[4, :, :]
    x = poloidata_3d[nfield, :, :]
    y = poloidata_3d[nfield+1, :, :]
    flux_3d = fluxdata.reshape((nfield, mtoroidal, mt_plus))
    phi_flux = flux_3d[0, :, :]
    phi_rms_psi = np.zeros(mpsi_plus)
    for ip in range(mpsi_plus):
        phi_rms_psi[ip] = np.sum(phi_2d[ip, :]**2) / mt_plus
    spec_pol = np.zeros(mt_plus)
    for itor in range(mtoroidal):
        spec_pol += np.abs(fftpack.fft(phi_flux[itor, :])) / mtoroidal
    spec_tor = np.zeros(mtoroidal)
    for ipol in range(mt_plus):
        spec_tor += np.abs(fftpack.fft(phi_flux[:, ipol])) / mt_plus
    ret = {"Status": 0, "file": base_filename,
           "phi_2d": phi_2d, "x": x, "y": y, "phi_flux": phi_flux,
           "step": step, "info": "", "phi_rms_psi": phi_rms_psi,
           "spec_pol": spec_pol, "spec_para": spec_tor,
           "apara_2d": apara_2d,"flowi": flowi, "densityi": densityi, "pressureipara":pressureipara,"pressureiperp":pressureiperp,"sfluidne":sfluidne}
    return ret