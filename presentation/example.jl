using Enzyme

function f(x)
    #  sin(x1*x2) * x2 + x1
    w1 = x[1] * x[2]
    w2 = x[2] * sin(w1)
    w3 = x[1] + w2
    return w3
end

function df_x1_sym(x)
    return 1 + x[2] * cos(x[1] * x[2]) * x[2]
end

x  = [2.0, 2.0]
dx = [1.0, 0.0]

dy_ad = Enzyme.autodiff(Forward, f, Duplicated(x,dx))[1]
println("With forward mode: f'(x) = $dy_ad")

dy_sym = df_x1_sym(x)
println("With symbolic method: f'(x) = $dy_sym")
