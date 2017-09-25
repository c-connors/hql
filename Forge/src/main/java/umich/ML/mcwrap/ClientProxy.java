package umich.ML.mcwrap;

public class ClientProxy extends CommonProxy {
	@Override
	public void RegisterRenders() 
	{
		CustomRender.register();
	}
}
