function [x, cutvalue, cutvalue_upperbound, Y] = run(seed,input_file_name,output_file_name)
    addpath(pwd);
    addpath(genpath(pwd));

    rng(str2double(seed));

    output = load(input_file_name);
    laplacian = double(output.inputs);
    L = diag(sum(laplacian,2)) - laplacian;
    [x, cutvalue, cutvalue_upperbound, Y, totaltime] = maxcut(L);
    save(output_file_name, 'x', 'cutvalue', 'cutvalue_upperbound', 'totaltime');
end

