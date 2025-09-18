# pylint: disable=W0212

import os
import numpy as np
import pyscf
import json

import opt_einsum as oe

from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
from pyscf.cc import ccsd_t_slow as ccsd_t
from pyscf.cc import ccsd_rdm
from pyscf.cc.ccsd_t_rdm_slow import _gamma1_intermediates
from pyscf.cc.ccsd_t_rdm_slow import _gamma2_intermediates
from pyscf.grad import ccsd_t as ccsd_t_grad, ccsd as ccsd_grad

from cc2cc.utils import diff_rho
from cc2cc.utils import DATA_PATH, AU2KCALMOL, ORCA_AVAILABLE
from cc2cc.utils.modelscf_rks import get_veff_grad_modified_zeros


def get_dft_energy(
    mol,
    grids,
    dm1_dft,
    e_dft,
    mdft,
    mf,
    dm1_cc,
    dm1_cc_mo,
    dm2_cc,
    e_cc,
    evaluate=False,
):
    """
    Calculate the (exchange-correlation energy - DFT energy) on the grids.
    """
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    ao_array = np.array([ao_value[0], ao_value[1], ao_value[2], ao_value[3]])
    ao_mat = np.array(
        [
            [ao_value[1], ao_value[2], ao_value[3]],
            [ao_value[4], ao_value[5], ao_value[6]],
            [ao_value[5], ao_value[7], ao_value[8]],
            [ao_value[6], ao_value[8], ao_value[9]],
        ]
    )
    ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
    ao_value = ao_value[:4]

    rho_dft = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft, xctype="GGA")
    rho_cc = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="GGA")

    ni = mdft._numint
    dft_mo_coeff = mdft.mo_coeff
    mf_mo_coeff = mf.mo_coeff

    vxc_lda = ni.eval_xc_eff("LDA,", rho_dft[0], deriv=1, xctype="LDA")[1]
    vxc_vwn = ni.eval_xc_eff(",VWN3", rho_dft[0], deriv=1, xctype="LDA")[1]
    vxc_b88 = ni.eval_xc_eff("B88,", rho_dft, deriv=1, xctype="GGA")[1]
    vxc_lyp = ni.eval_xc_eff(",LYP", rho_dft, deriv=1, xctype="GGA")[1]

    vxc_b3lyp = np.zeros((4, 4, len(grids.coords)))
    vxc_b3lyp[0, 0:1, :] = vxc_lda
    vxc_b3lyp[1, 0:1, :] = vxc_vwn
    vxc_b3lyp[2, :, :] = vxc_b88
    vxc_b3lyp[3, :, :] = vxc_lyp

    wv = grids.weights * vxc_b3lyp
    wv[:, 0, :] *= 0.5

    atmlst = range(mol.natm)
    grad2force = np.zeros((len(atmlst), 4, len(grids.coords), 3))
    for k, ia in enumerate(atmlst):
        p0, p1 = mol.aoslice_by_atom()[ia, 2:]
        grad2force[k] = np.einsum(
            "mnp,xpi,npj,ij->mpx",
            wv,
            ao_value[1:4, :, p0:p1],
            ao_array,
            dm1_dft[p0:p1],
            optimize=True,
        ) + np.einsum(
            "mnp,nxpi,pj,ij->mpx",
            wv,
            ao_mat[:, :, :, p0:p1],
            ao_value[0],
            dm1_dft[p0:p1],
            optimize=True,
        )
    grad2force = -grad2force * 2

    if evaluate:
        return None, None, rho_cc, rho_dft, grad2force
    else:
        dm12 = (
            0.5 * dm2_cc
            - 0.5 * oe.contract("pq,rs->pqrs", dm1_dft, dm1_dft)
            + 0.05 * oe.contract("pr,qs->pqrs", dm1_dft, dm1_dft)
        )
        # exchange part
        # + 0.5 * alpha * oe.contract("pr,qs->pqrs", dm1_cc * 0.5, dm1_cc * 0.5)
        # + 0.5 * alpha * oe.contract("pr,qs->pqrs", dm1_cc * 0.5, dm1_cc * 0.5)
        # alpha is 0.2 in b3lyp

        expr_rinv_dm2_r = oe.contract_expression(
            "ijkl,i,j,kl->",
            dm12,
            (mol.nao,),
            (mol.nao,),
            (mol.nao, mol.nao),
            constants=[0],
            optimize="optimal",
        )

        exc_cc_grids = np.zeros_like(rho_dft[0])

        for i, coord in enumerate(grids.coords):
            if i * 10 % len(grids.coords) == 0:
                print(f"Progress: {(i*100)/len(grids.coords):.1f}%", flush=True)
            with mol.with_rinv_origin(coord):
                rinv = mol.intor("int1e_rinv")
                exc_cc_grids[i] += expr_rinv_dm2_r(
                    ao_value[0][i],
                    ao_value[0][i],
                    rinv,
                    backend="torch",
                )

        eri = mol.intor("int2e")
        error_energy = oe.contract("pqrs,pqrs->", eri, dm12)
        error = np.sum(exc_cc_grids * grids.weights) - error_energy
        print(
            "exc_cc_grids: ",
            f"error_energy: {AU2KCALMOL * error_energy},",
            f"Error: {AU2KCALMOL * error},",
        )

        # kinetic part
        eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(dm1_cc_mo)
        eigs_v_dm1 = mf_mo_coeff @ eigs_v_dm1
        for i in range(np.shape(eigs_v_dm1)[1]):
            part = oe.contract(
                "pm,m,n,pn->p",
                ao_value[0],
                eigs_v_dm1[:, i],
                eigs_v_dm1[:, i],
                ao_2_diag,
            )
            exc_cc_grids -= part * eigs_e_dm1[i] / 2

        for i in range(mol.nelec[0]):
            part = oe.contract(
                "pm,m,n,pn->p",
                ao_value[0],
                dft_mo_coeff[:, i],
                dft_mo_coeff[:, i],
                ao_2_diag,
            )
            exc_cc_grids += part

        kin = mol.intor("int1e_kin")
        error_energy += oe.contract("pq,pq->", kin, dm1_cc - dm1_dft)
        error = np.sum(exc_cc_grids * grids.weights) - error_energy
        print(
            "exc_cc_grids: ",
            f"error_energy: {AU2KCALMOL * error_energy},",
            f"Error: {AU2KCALMOL * error},",
        )

        for i, coord in enumerate(grids.coords):
            for i_atom in range(mol.natm):
                distance = np.linalg.norm(mol.atom_coords()[i_atom] - coord)
                if distance > 1e-3:
                    exc_cc_grids[i] -= (
                        (rho_cc[0][i] - rho_dft[0][i])
                        * mol.atom_charges()[i_atom]
                        / distance
                    )

        nuc = mol.intor("int1e_nuc")
        error_energy += oe.contract("pq,pq->", nuc, dm1_cc - dm1_dft)
        error = np.sum(exc_cc_grids * grids.weights) - error_energy
        print(
            "exc_cc_grids: ",
            f"error_energy: {AU2KCALMOL * error_energy},",
            f"Error: {AU2KCALMOL * error},",
        )

        # for i, coord in enumerate(grids.coords):
        #     for i_atom in range(mol.natm):
        #         distance = np.linalg.norm(mol.atom_coords()[i_atom] - coord)
        #         if distance > 1e-3:
        #             exc_cc_grids[i] -= (
        #                 (rho_cc[0][i] - rho_dft[0][i])
        #                 * mol.atom_charges()[i_atom]
        #                 / distance
        #             )

        ecp = mol.intor("ECPscalar")
        error_energy += oe.contract("pq,pq->", ecp, dm1_cc - dm1_dft)
        error = np.sum(exc_cc_grids * grids.weights) - error_energy
        print(
            "exc_cc_grids: ",
            f"error_energy: {AU2KCALMOL * error_energy},",
            f"Error: {AU2KCALMOL * error},",
        )

        exc_cc_grids -= pyscf.dft.libxc.eval_xc("b3lyp", rho_dft)[0] * rho_dft[0]
        error_energy += -np.sum(
            pyscf.dft.libxc.eval_xc("b3lyp", rho_dft)[0] * rho_dft[0] * grids.weights
        )
        error = np.sum(exc_cc_grids * grids.weights) - error_energy
        print(
            "exc_cc_grids: ",
            f"error_energy: {AU2KCALMOL * error_energy},",
            f"Error: {AU2KCALMOL * error},",
        )

        print("Final result:")
        print(f"e_cc: {e_cc}, e_dft: {e_dft}")
        print(
            f"e_cc: {oe.contract("pq,pq->", ecp + kin + nuc, dm1_cc) + 0.5 * oe.contract("pqrs,pqrs->", eri, dm2_cc) + mol.energy_nuc()}, e_dft: {oe.contract("pq,pq->", ecp + kin + nuc, dm1_dft) + 0.5 * oe.contract("pqrs,pq,rs->", eri, dm1_dft, dm1_dft) - 0.05 * oe.contract("pqrs,pr,qs->", eri, dm1_dft, dm1_dft) + np.sum(pyscf.dft.libxc.eval_xc("b3lyp", rho_dft)[0] * rho_dft[0] * grids.weights) + mol.energy_nuc()}"
        )
        error_energy = e_cc - e_dft
        error = np.sum(exc_cc_grids * grids.weights) - error_energy
        print(
            "exc_cc_grids: ",
            f"max exc_cc_grids: {np.max(exc_cc_grids)}",
            f"min exc_cc_grids: {np.min(exc_cc_grids)}",
            f"mean exc_cc_grids: {np.mean(exc_cc_grids)}",
            f"std exc_cc_grids: {np.std(exc_cc_grids)}",
            f"error_energy: {AU2KCALMOL * error_energy},",
            f"Error: {AU2KCALMOL * error},",
        )
        return error_energy, exc_cc_grids, rho_cc, rho_dft, grad2force


def cc(mol, grids, name, args, evaluate=False):
    """
    Generate data for the CCSD method. (Restrict scenario to spin 0).
    """

    print(f"Generate data for {name}")
    # RHF calculation
    mf = pyscf.scf.RHF(mol).newton()
    mf.max_cycle = 200
    mf.kernel()
    if args.check_convergence and not mf.converged:
        raise ValueError("RHF not converged.")

    # DFT calculation
    mdft = pyscf.scf.RKS(mol)
    mdft.verbose = 4
    mdft.max_cycle = 200
    mdft.xc = "b3lyp"
    mdft.kernel()
    if args.check_convergence and not mdft.converged:
        raise ValueError("RKS not converged.")
    dm1_dft = mdft.make_rdm1(ao_repr=True)
    e_dft = mdft.e_tot
    gdft = mdft.Gradients()
    grad_dft = gdft.kernel()

    # CCSD calculation

    if evaluate:
        if ORCA_AVAILABLE:
            print("Use ORCA to evaluate CCSD(T)")
            maxcore = 2000  # in MB (each core! not total)
            molecular_xyz = ""
            for atom_info in mol._atom:
                molecular_xyz += (
                    f"{atom_info[0]:<6}\t{atom_info[1][0]:<16.10}\t{atom_info[1][1]:<16.10}\t{atom_info[1][2]:<16.10}"
                    + "\n"
                )

            if not os.path.exists(f"tmp_mol/{name}"):
                os.makedirs(f"tmp_mol/{name}")

            with open(f"tmp_mol/{name}/mol.inp", "w", encoding="utf-8") as f:
                f.write(
                    f"""! {args.basis} CCSD(T) TightSCF PrintBasis

                %method
                    WriteJSONPropertyfile True
                    FrozenCore FC_NONE
                end

                %maxcore {maxcore}

                %MDCI
                    MaxCore {maxcore}
                end

                %pal
                    nprocs {os.environ.get("OMP_NUM_THREADS")}
                end

                %coords
                CTyp   xyz     # the type of coordinates = xyz or internal
                Charge {mol.charge}       # the total charge of the molecule
                Mult   {mol.spin+1}        # the multiplicity = 2S+1
                Units  bohrs    # the unit of length = angs or bohrs

                # the subblock coords is for the actual coordinates
                # for CTyp=xyz
                coords
                {molecular_xyz}end
                end
                """
                )

            os.system(f"$(which orca) tmp_mol/{name}/mol.inp > tmp_mol/{name}/mol.out")

            if not (os.path.exists(f"tmp_mol/{name}/mol.property.json")):
                print("ORCA calculation failed, no JSON file found.")
                # # Clear the directory if it already exists to avoid disk space issues
                for file in os.listdir(f"tmp_mol/{name}"):
                    os.remove(os.path.join(f"tmp_mol/{name}", file))
                raise ValueError("ORCA calculation failed, no JSON file found.")

            with open(f"tmp_mol/{name}/mol.property.json", "r", encoding="UTF-8") as f:
                data_json = json.load(f)
                print(data_json)

            e_cc = data_json["Geometry_1"]["MDCI_Energies"]["TOTALENERGY"]
            grad_cc = np.zeros((mol.natm, 3))
            dm1_cc = None
        else:
            mycc = pyscf.cc.CCSD(mf)
            mycc.verbose = 4
            mycc.direct = True
            _, t1, t2 = mycc.kernel()
            eris = mycc.ao2mo()
            e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
            l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)[1:]
            dm1_cc = ccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris=eris, ao_repr=True)
            cc_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_cc, unit="A.U.")
            print(f"CCSD dipole: {cc_dipole}")
            del t1, t2, l1, l2
            e_cc = mycc.e_tot + e3ref
            print(f"CCSD(T) energy: {e_cc}")
            if mol.natm == 1:
                grad_cc = np.zeros((mol.natm, 3))
            else:
                gcc = ccsd_t_grad.Gradients(mycc)
                grad_cc = gcc.kernel()

        dm1_cc_mo = None
        dm2_cc = None
    else:
        mycc = pyscf.cc.CCSD(mf)
        mycc.verbose = 4
        _, t1, t2 = mycc.kernel()
        if args.cc_triple:
            eris = mycc.ao2mo()
            e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
            l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)[1:]
            dm1_cc = ccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris=eris, ao_repr=True)
            dm1_cc_mo = ccsd_t_rdm.make_rdm1(
                mycc, t1, t2, l1, l2, eris=eris, ao_repr=False
            )
            d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
            d2 = _gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
            dm2_cc = ccsd_rdm._make_rdm2(mycc, d1, d2, True, True, ao_repr=True)
            del t1, t2, l1, l2, d1, d2
            e_cc = mycc.e_tot + e3ref
            print(f"CCSD(T) energy: {e_cc}")
            if mol.natm == 1:
                grad_cc = np.zeros((mol.natm, 3))
            else:
                gcc = ccsd_t_grad.Gradients(mycc)
                grad_cc = gcc.kernel()
        else:
            dm1_cc = mycc.make_rdm1(ao_repr=True)
            dm1_cc_mo = mycc.make_rdm1(ao_repr=False)
            dm2_cc = mycc.make_rdm2(ao_repr=True)
            e_cc = mycc.e_tot
            if mol.natm == 1:
                grad_cc = np.zeros((mol.natm, 3))
            else:
                gcc = ccsd_grad.Gradients(mycc)
                grad_cc = gcc.kernel()

        print(f"{diff_rho(mol, dm1_cc, dm1_dft, grids):.6f} (CCSD vs DFT)")
        cc_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_cc, unit="A.U.")
        dft_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_dft, unit="A.U.")
        print(f"{np.linalg.norm(cc_dipole - dft_dipole)} (CCSD vs DFT)")

    # Compare CCSD and DFT
    energy_train = e_cc - e_dft
    grad_cc_train = grad_cc - grad_dft

    # Calculate the (exchange-correlation energy - DFT energy) on the grids and the grad to force matrix
    error_energy_dft, exc_cc_grids_dft, rho_cc, rho_dft, grad2force = get_dft_energy(
        mol,
        grids,
        dm1_dft,
        e_dft,
        mdft,
        mf,
        dm1_cc,
        dm1_cc_mo,
        dm2_cc,
        e_cc,
        evaluate=evaluate,
    )

    # Test force
    grad_mat = np.array(
        [
            0.08 * np.ones(len(grids.coords)),
            0.19 * np.ones(len(grids.coords)),
            0.72 * np.ones(len(grids.coords)),
            0.81 * np.ones(len(grids.coords)),
        ]
    )
    force = np.einsum(
        "mp,impx->ix",
        grad_mat,
        grad2force,
        optimize=True,
    )
    get_veff_grad_modified_zeros(gdft)
    grad_dft_zeros = gdft.kernel()
    print("Error force DFT: ", np.linalg.norm(force - (grad_dft - grad_dft_zeros)))

    # Generate input data
    rho_cube_cc = grids.gen_cube_rho_rks(rho_cc, mdft._numint, dm1_cc)
    rho_cube_dft = grids.gen_cube_rho_rks(rho_dft, mdft._numint, dm1_dft)
    np.savez_compressed(
        DATA_PATH / f"data_{name}.npz",
        mol=mol.tostring(format="xyz"),
        charge=mol.charge,
        spin=mol.spin,
        e_cc=e_cc,
        energy_train=energy_train,
        dm1_cc=dm1_cc,
        rho_cube_cc=rho_cube_cc,
        rho_cube_dft=rho_cube_dft,
        weights=grids.weights,
        exc_cc_grids=exc_cc_grids_dft,
        error_energy=error_energy_dft,
        grad2force=grad2force,
        grad_cc_train=grad_cc_train,
    )
