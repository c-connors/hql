package umich.ML.mcwrap.configuration;

import net.minecraftforge.common.config.Configuration;
import net.minecraftforge.fml.common.FMLCommonHandler;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.NodeList;
import umich.ML.mcwrap.util.FileParser;

import java.io.File;
import java.lang.reflect.Field;

/**************************************************
 * Package: umich.ML.mcwrap.configuration
 * Class: ConfigurationHandler
 * Timestamp: 3:52 PM 12/19/15
 * Authors: Valliappa Chockalingam, Junhyuk Oh
 **************************************************/

// NOTE: Singleton Class (zero or one instances are present at any given time)
public class ConfigurationHandler {
    private static Configuration configuration = null;

    private static int port;

    private static int screenHeight;

    private static int screenWidth;

    private static Boolean log;

    private static int terminateAfter;

    private static Boolean roofEnable;

    private static int mapUpdateFreq;

    private static int logFreq;
    
    private static int map_size;
    
    private static int object_y;
    
    private static int player_y;

    private ConfigurationHandler() {}

    private static Boolean initialized = false;

    // REQUIRES: configFile to be set to the file containing the configuration
    // MODIFIES: configuration and all private static variables associated with this class
    // EFFECTS:  loads configuration file and hence the variables associated with it
    // NOTE: Incorrect file could lead to simply using defaults
    public static void init(File configFile)
    {
        // If the configuration instance is assigned, do nothing
        //if(!initialized)
        //{
            System.out.print("[[ Configuration Handler  (( init )) ]] : No singleton, " +
                    "creating an instance for the first time! Proceeding.\n");

            configuration = new Configuration(configFile);

            port = configuration.get("Network", "port", 0,
                    "Choose Port No. Random port is chosen by default (0).").getInt();
            
            screenHeight = configuration.get("Screen Dimension", "height", 32, "Screen Height").getInt();

            screenWidth = configuration.get("Screen Dimension", "width", 32, "Screen Width").getInt();

            //log = configuration.get("Debug", "log", true, "Log useful info.").getBoolean();
            //logFreq = configuration.get("Debug", "logFreq", 1000, "Log frequency.").getInt();

            roofEnable = configuration.get("Task", "roofEnable", true, "Enable Roof?").getBoolean();

            if(configuration.hasChanged())
            {
                System.out.print("[[ Configuration Handler (( init )) ]] : Configuration has changed. Saving.\n");
                configuration.save();
                System.out.print("[[ Configuration Handler (( init )) ]] : Completed save.\n");
            }

            initialized = true;
        //}

        //else System.out.print("[[ Configuration Handler (( init )) ]] : Singleton exists, " +
        //        "skipping init.\n");
    }

    /* Getters for configuration variables */

    public static int getPort() {
        return port;
    }
    
    public static int getScreenHeight() {
        return screenHeight;
    }

    public static int getScreenWidth() {
        return screenWidth;
    }

    public static void setScreenRes(String screenRes)
    {
        screenHeight = Integer.parseInt(screenRes.replace("\n", "").split(" ")[1]);
        screenWidth = Integer.parseInt(screenRes.replace("\n", "").split(" ")[2]);
    }

    public static Boolean getLog() {
        return log;
    }

    public static int getTerminateAfter() {
        return terminateAfter;
    }

    public static void initTask(String task_xml)
    {
        Element root = FileParser.readXML(task_xml);
        NodeList nList = root.getElementsByTagName("map_size");
        NamedNodeMap namedNodeMap = nList.item(0).getAttributes();
        map_size = Integer.parseInt(namedNodeMap.getNamedItem("x").getNodeValue());
        
        nList = root.getElementsByTagName("y_axis");
        namedNodeMap = nList.item(0).getAttributes();
        object_y = Integer.parseInt(namedNodeMap.getNamedItem("object").getNodeValue());
        player_y = Integer.parseInt(namedNodeMap.getNamedItem("player").getNodeValue());
    }

    public static int getMapSize() {
    	return map_size;
    }
    public static int getPlayerY() {
    	return player_y;
    }
    public static int getObjectY() {
    	return object_y;
    }
    
    public static int getMapUpdateFreq() {
        return mapUpdateFreq;
    }

    public static Boolean getRoofEnable() {
        return roofEnable;
    }

    public static int getLogFreq() { return logFreq; }
  

    public static void printInfo()
    {
        if(!initialized) {
            System.out.print("[[ Configuration Handler (( printInfo )) ]] : No singleton exists, " +
                    "cannot proceed in printing info. Aborting!\n");
            FMLCommonHandler.instance().exitJava(-1, false);
        }

        else
        {
            System.out.print("\n\n\n");

            for(Field field : ConfigurationHandler.class.getDeclaredFields())
            {
                try {
                    System.out.print(field.getName() + " : " + field.get(ConfigurationHandler.class) + "\n");
                }

                catch(IllegalAccessException e)
                {
                    FMLCommonHandler.instance().exitJava(-1, false);
                }
            }

            System.out.print("\n\n\n");
        }
    }
}
