import ucar.nc2.NetcdfFile;

import java.io.IOException;

public class Pipeline  {
    //private static final String PATH = System.getenv("HOME2") + "/datasets/ECAD/original/small_sample/";
    private static final String PATH = "/export/scratch1/radujica/datasets/ECAD/original/small_sample/";

    public void start() throws IOException {
        NetcdfFile f = NetcdfFile.open(PATH + "data1.nc");
        System.out.println(f.getVariables());
    }
}
