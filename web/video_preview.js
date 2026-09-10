import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

export function omitVideoImagePreview(route) {
  if (typeof route !== "string" || !route.startsWith("/view?")) return route;
  const query = new URLSearchParams(route.slice(6));
  const filename = query.get("filename") || "";
  if (!query.has("preview") || !/\.(mp4|webm|mkv|mov|avi|m4v|mpeg|mpg)$/i.test(filename)) return route;
  query.delete("preview");
  return `/view?${query}`;
}

app.registerExtension({
  name: "UtilsCollection.VideoPreview",
  setup() {
    const originalApiURL = api.apiURL;
    api.apiURL = function (route) {
      return originalApiURL.call(this, omitVideoImagePreview(route));
    };
  },
});
