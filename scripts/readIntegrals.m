function [tasks, integrals, errors] = readIntegrals(filename, sep = ';')
  data = dlmread(filename, sep);
  tasks = data(2:end, 1:2);
  integrals = data(2:end, 3:5);

  if size(data)(2) == 6
    errors = data(2:end, 6);
  end
end
