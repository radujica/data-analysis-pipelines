import ucar.nc2.Attribute;
import ucar.nc2.NetcdfFile;
import ucar.nc2.Variable;
import ucar.nc2.time.CalendarDate;

import java.io.IOException;

public class Pipeline  {
    //private static final String PATH = System.getenv("HOME2") + "/datasets/ECAD/original/small_sample/";
    private static final String PATH = "/export/scratch1/radujica/datasets/ECAD/original/small_sample/";
    private static final String CALENDAR = "proleptic_gregorian";
    private static final String UNITS = "days since 1950-01-01";

    private Object readVariable(Variable variable) throws IOException {
        Class<?> dataType = variable.getDataType().getPrimitiveClassType();
        Object data;

        Attribute scaleFactorAttribute = variable.findAttribute("scale_factor");
        if (scaleFactorAttribute != null) {
            // TODO: should probably be more generic
            float scaleFactor = scaleFactorAttribute.getNumericValue().floatValue();
            float[] dataRaw = (float[]) variable.read().get1DJavaArray(float.class);
            for (int i = 0; i < dataRaw.length; i++) {
                dataRaw[i] *= scaleFactor;
            }
            data = dataRaw;
        } else {
            data = variable.read().get1DJavaArray(dataType);
            // time is handled manually later, so keep as int[]
        }

        return data;
    }

    // Time kept as original int so use this method to convert to string
    private String intTimeToString(int time) {
        String udunits = String.valueOf(time) + " " + UNITS;

        return CalendarDate.parseUdunits(CALENDAR, udunits).toString().substring(0, 10);
    }

    private DataFrame readData(String path) throws IOException {
        NetcdfFile file = NetcdfFile.open(path);

        DataFrame df = new DataFrame();
        for (Variable variable : file.getVariables()) {
            df.add(variable.getShortName(), readVariable(variable));
        }

        return df;
    }

    public void start() throws IOException {
        DataFrame df1 = readData(PATH + "data1.nc");
        DataFrame df2 = readData(PATH + "data2.nc");

        DataFrame df = df1.join(df2);

        float[] tg = (float[]) df.get("tg");
    }
}
