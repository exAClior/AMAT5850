using Molly
using Zygote
using Format
using LinearAlgebra

dist_true = 1.0

# change information
# make it simulate water
# try GPU acceleration?
n_steps = 150
atom_mass = 10.0
boundary = CubicBoundary(3.0)
temp = 0.05
coords = [
    SVector(0.8, 0.75, 1.5), SVector(1.5, 0.70, 1.5), SVector(2.3, 0.75, 1.5),
    SVector(0.8, 2.25, 1.5), SVector(1.5, 2.20, 1.5), SVector(2.3, 2.25, 1.5),
]
n_atoms = length(coords)
velocities = zero(coords)
simulator = VelocityVerlet(
    dt=0.05,
    coupling=BerendsenThermostat(temp, 0.5),
)

function loss(θ)
    atoms = [Atom(0, 0.0, atom_mass, 0.0, 0.0, false) for i in 1:n_atoms]
    loggers = (coords=CoordinateLogger(Float64, 2),)
    specific_inter_lists = (
        InteractionList2Atoms(
            [1, 2, 4, 5],
            [2, 3, 5, 6],
            [HarmonicBond(100.0, 0.7) for _ in 1:4],
        ),
        InteractionList3Atoms(
            [1, 4],
            [2, 5],
            [3, 6],
            [HarmonicAngle(10.0, θ), HarmonicAngle(10.0, θ)],
        ),
    )

    s = System(
        atoms=atoms,
        specific_inter_lists=specific_inter_lists,
        coords=deepcopy(coords),
        velocities=deepcopy(velocities),
        boundary=boundary,
        loggers=loggers,
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    simulate!(s, simulator, n_steps)

    d1 = norm(vector(s.coords[1], s.coords[3], boundary))
    d2 = norm(vector(s.coords[4], s.coords[6], boundary))
    dist_end = 0.5 * (d1 + d2)
    loss_val = abs(dist_end - dist_true)

    Zygote.ignore() do
        printfmt("θ {:5.1f}°  |  Final dist {:4.2f}  |  Loss {:5.3f}  |  ",
                 rad2deg(θ), dist_end, loss_val)
    end

    return loss_val
end

function train()
    θlearn = deg2rad(110.0)
    n_epochs = 50

    for epoch_n in 1:n_epochs
        printfmt("Epoch {:>2}  |  ", epoch_n)
        grad = gradient(loss, θlearn)[1]
        printfmt("Grad {:6.3f}\n", round(grad; digits=2))
        θlearn -= grad * 0.1
    end
end

train()
