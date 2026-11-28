"""VV10 energy-density inputs and their fixed-density directional derivatives."""

import numpy as np
from pyscf.dft import numint


def _vv10_density_jvp(rho, coords, weights, drho, dcoords, dweights, pars):
    """Differentiate the unweighted VV10 energy density on a common grid."""
    active = rho[0] >= 1e-8
    result = np.zeros((rho.shape[1], drho.shape[2]), dtype=np.float64)
    if not np.any(active):
        return result

    r = rho[0, active]
    grad = rho[1:, active]
    xyz = coords[active]
    w = weights[active]
    dr = drho[0, active]
    dgrad = drho[1:, active]
    dxyz = dcoords[active]
    dw = dweights[active]

    b, c = pars
    pi43 = 4 * np.pi / 3
    beta = (3 / b**2) ** 0.75 / 32
    gradient_squared = np.sum(grad**2, axis=0)
    dgradient_squared = 2 * np.einsum("ap,apd->pd", grad, dgrad)
    omega = np.sqrt(c * gradient_squared**2 / r**4 + pi43 * r)
    domega = (
        c
        * (
            2 * gradient_squared[:, None] * dgradient_squared / r[:, None] ** 4
            - 4 * gradient_squared[:, None] ** 2 * dr / r[:, None] ** 5
        )
        + pi43 * dr
    ) / (2 * omega[:, None])
    kappa = b * 1.5 * np.pi * (9 * np.pi) ** (-1 / 6) * r ** (1 / 6)
    dkappa = kappa[:, None] * dr / (6 * r[:, None])

    inner_weight_response = w[:, None] * dr + r[:, None] * dw
    active_result = np.zeros((len(r), dr.shape[1]), dtype=np.float64)
    for start in range(0, len(r), 128):
        stop = min(start + 128, len(r))
        total = np.zeros(stop - start, dtype=np.float64)
        local_kappa = np.zeros(stop - start, dtype=np.float64)
        local_omega = np.zeros(stop - start, dtype=np.float64)
        derivative = np.zeros((stop - start, dr.shape[1]), dtype=np.float64)
        for inner_start in range(0, len(r), 2048):
            inner_stop = min(inner_start + 2048, len(r))
            inner = slice(inner_start, inner_stop)
            delta = xyz[start:stop, None, :] - xyz[None, inner, :]
            distance2 = np.sum(delta**2, axis=-1)
            g = distance2 * omega[start:stop, None] + kappa[start:stop, None]
            gp = distance2 * omega[None, inner] + kappa[None, inner]
            inv = 1 / (g * gp * (g + gp))
            term = inv * (r[inner] * w[inner])[None, :]
            outer_factor = term * (1 / g + 1 / (g + gp))
            inner_factor = term * (1 / gp + 1 / (g + gp))

            total += term.sum(axis=1)
            local_kappa += outer_factor.sum(axis=1)
            local_omega += (outer_factor * distance2).sum(axis=1)
            derivative += inv @ inner_weight_response[inner]
            derivative -= inner_factor @ dkappa[inner]
            derivative -= (inner_factor * distance2) @ domega[inner]

            distance_factor = 2 * (
                outer_factor * omega[start:stop, None]
                + inner_factor * omega[None, inner]
            )
            for axis in range(3):
                position_factor = distance_factor * delta[:, :, axis]
                derivative += position_factor @ dxyz[inner, axis]
                derivative -= (
                    position_factor.sum(axis=1)[:, None]
                    * dxyz[start:stop, axis]
                )

        derivative -= local_kappa[:, None] * dkappa[start:stop]
        derivative -= local_omega[:, None] * domega[start:stop]
        active_result[start:stop] = (
            beta * dr[start:stop]
            - 0.75
            * (
                total[:, None] * dr[start:stop]
                + r[start:stop, None] * derivative
            )
        )
    result[active] = active_result
    return result


def vv10_input_data(
    mol,
    coords,
    weights,
    dm,
    *,
    unrestricted,
    weight_response=None,
    grid_owners=None,
    projection_dms=(),
):
    """Return VV10 density, nuclear Jacobian, and projected density Jacobians.

    Nuclear derivatives keep the AO density matrix fixed and include motion of
    both quadrature points and their weights. Projection matrices are symmetric
    AO density perturbations, one array of shape (directions, nao, nao) per spin.
    """
    coords = np.asarray(coords, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    dm = np.asarray(dm, dtype=np.float64)
    dm_total = dm.sum(axis=0) if unrestricted else dm
    ngrid = len(weights)
    if coords.shape != (ngrid, 3):
        raise ValueError("VV10 coordinates and weights have different grid sizes.")

    force_directions = 3 * mol.natm if weight_response is not None else 0
    projection_dms = tuple(np.asarray(x, dtype=np.float64) for x in projection_dms)
    projection_sizes = [len(x) for x in projection_dms]
    ndirection = force_directions + sum(projection_sizes)
    ni = numint.NumInt()
    rho = np.zeros((4, ngrid), dtype=np.float64)
    density = np.zeros(ngrid, dtype=np.float64)
    derivative = np.zeros((ngrid, ndirection), dtype=np.float64)

    drho = np.zeros((4, ngrid, ndirection), dtype=np.float64)
    dcoords = np.zeros((ngrid, 3, ndirection), dtype=np.float64)
    dweights = np.zeros((ngrid, ndirection), dtype=np.float64)
    if force_directions:
        grid_owners = np.asarray(grid_owners)
        weight_response = np.asarray(weight_response, dtype=np.float64)
        if grid_owners.shape != (ngrid,) or weight_response.shape != (
            mol.natm, 3, ngrid
        ):
            raise ValueError("VV10 nuclear response requires aligned grid mappings.")
        if np.any((grid_owners < -1) | (grid_owners >= mol.natm)):
            raise ValueError("VV10 grid owners must identify an atom or fixed padding.")
        dweights[:, :force_directions] = weight_response.transpose(2, 0, 1).reshape(
            ngrid, force_directions
        )
        for atom in range(mol.natm):
            for axis in range(3):
                dcoords[grid_owners == atom, axis, 3 * atom + axis] = 1

    second = ((4, 5, 6), (5, 7, 8), (6, 8, 9))
    for start in range(0, ngrid, 2048):
        stop = min(start + 2048, ngrid)
        ao = ni.eval_ao(mol, coords[start:stop], deriv=2 if force_directions else 1)
        rho_block = ni.eval_rho(mol, ao[:4], dm_total, xctype="GGA", hermi=1)
        rho[:, start:stop] = rho_block
        drho_block = drho[:, start:stop]
        if force_directions:
            ao_dm = np.einsum("xpi,ij->xpj", ao[:4], dm_total)
            for atom, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
                moved = grid_owners[start:stop] == atom
                for axis in range(3):
                    direction = 3 * atom + axis
                    drho_block[0, :, direction] = -2 * np.einsum(
                        "pi,pi->p", ao[axis + 1, :, p0:p1], ao_dm[0, :, p0:p1]
                    )
                    drho_block[0, moved, direction] += rho_block[axis + 1, moved]
                    for component in range(3):
                        second_ao = ao[second[component][axis]]
                        drho_block[component + 1, :, direction] = -2 * (
                            np.einsum(
                                "pi,pi->p",
                                second_ao[:, p0:p1],
                                ao_dm[0, :, p0:p1],
                            )
                            + np.einsum(
                                "pi,pi->p",
                                ao[axis + 1, :, p0:p1],
                                ao_dm[component + 1, :, p0:p1],
                            )
                        )
                        drho_block[component + 1, moved, direction] += 2 * (
                            np.einsum("pi,pi->p", second_ao[moved], ao_dm[0, moved])
                            + np.einsum(
                                "pi,pi->p",
                                ao[axis + 1, moved],
                                ao_dm[component + 1, moved],
                            )
                        )

        offset = force_directions
        for projected_dm in projection_dms:
            end = offset + len(projected_dm)
            for component in range(4):
                drho_block[component, :, offset:end] = (
                    (1 if component == 0 else 2)
                    * np.einsum(
                        "pi,kij,pj->pk",
                        ao[component],
                        projected_dm,
                        ao[0],
                        optimize=True,
                    )
                )
            offset = end

    for pars, factor in ni.nlc_coeff("vv10"):
        exc, _ = numint._vv10nlc(rho, coords, rho, weights, coords, pars)
        density += factor * rho[0] * exc
        if ndirection:
            derivative += factor * _vv10_density_jvp(
                rho, coords, weights, drho, dcoords, dweights, pars
            )

    forces = derivative[:, :force_directions] if force_directions else None
    projections = []
    offset = force_directions
    for size in projection_sizes:
        projections.append(derivative[:, offset:offset + size])
        offset += size
    return density, forces, projections
